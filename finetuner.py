import torch
import logging
import os, json
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PrefixTuningConfig, PromptEncoderConfig
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(self, config):
        """
        Fine-tuning module for classification tasks with PEFT and AutoConfig.

        Args:
            config (FineTunerArguments): Configuration object with all required settings.
        """
        self.config = config
        self.model_name = config.finetuner_model_name_or_path
        self.tokenizer_name = config.finetuner_tokenizer_name_or_path if config.finetuner_tokenizer_name_or_path else self.model_name
        self.fine_tune_method = config.fine_tune_method

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Configure and load the model
        model_config = AutoConfig.from_pretrained(self.model_name, num_labels=config.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=model_config)

        # Apply PEFT if specified
        if self.fine_tune_method in ["lora", "prefix_tuning", "soft_prompt", "compactor"]:
            self._apply_peft(self.fine_tune_method, config.peft_config)

    def _apply_peft(self, fine_tune_method, peft_config):
        """
        Apply Parameter-Efficient Fine-Tuning (PEFT) to the model.
        """
        logger.info(f"Applying {fine_tune_method} fine-tuning.")
        peft_config['task_type'] = 'SEQ_CLS'
        if fine_tune_method == "lora":
            config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        elif fine_tune_method == "prefix_tuning":
            config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif fine_tune_method == "soft_prompt":
            config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        else:
            raise ValueError(f"Unsupported PEFT method: {fine_tune_method}")
        self.model = get_peft_model(self.model, config)

    def prepare_data(self, data):
        """
        Prepare data from the DataProvider for training and evaluation.
        Args:
            data (Dataset): Hugging Face dataset with text and labels.
        Returns:
            dict: Tokenized dataset ready for training.
        """
        # Tokenize the dataset using the provided tokenizer
        encodings = self.tokenizer(data["text"], truncation=True, padding=True, max_length=self.config.max_seq_length)
        dataset = Dataset.from_dict(encodings)
        dataset = dataset.add_column("label", data["label"])
        return dataset

    def calculate_class_weights(self, labels):
        """
        Calculate class weights for imbalanced datasets.
        """
        unique_labels = np.array(list(set(labels)))
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return torch.tensor(class_weights, dtype=torch.float)

    def train(self, train_data, eval_data):
        """
        Train the model with weighted loss and metadata considerations.
        """
        logger.info("Starting training process.")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move weights to the appropriate device
        self.model.to(device)

        if self.config.use_class_weights:

            logger.info("** USING WEIGHTED LOSS**")
            if len(pd.unique(train_data["label"])) < 2:  # if a small sample only has examples of one class, we cannot do weighting
                class_weights = self.calculate_class_weights(eval_data["label"])
            else:
                class_weights = self.calculate_class_weights(train_data["label"])

            logger.info(f"class weights: {class_weights}")
            class_weights = class_weights.to(device)

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.get("labels")
                    # forward pass
                    outputs = model(**inputs)
                    logits = outputs.get("logits")
                    # compute custom loss
                    weighted_loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss = weighted_loss_fct(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            self.trainer = WeightedTrainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics
            )

        else:

            logger.info("** USING STANDARD UNWEIGHTED LOSS**")

            self.trainer = Trainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics
            )

        # # Create Trainer instance
        # self.trainer = Trainer(
        #     model=self.model,
        #     args=self.config,
        #     train_dataset=train_data,
        #     eval_dataset=eval_data,
        #     compute_metrics=self.compute_metrics
        # )

        self.trainer.train()

    def predict(self, test_data, save_prediction=False):
        """
        Predict labels for the test dataset.
        """
        logger.info("Running predictions.")
        # trainer = Trainer(model=self.model)
        predictions = self.trainer.predict(test_data)
        # Extract predictions and labels
        logits = predictions.predictions
        labels = predictions.label_ids
        if save_prediction:
            # Save predictions
            output_dir = self.config.output_dir
            predictions_path = os.path.join(output_dir, "predictions.txt")
            with open(predictions_path, "w") as f:
                f.write("Predicted\tTrue\n")
                predicted_labels = np.argmax(logits, axis=-1)
                for pred, label in zip(predicted_labels, labels):
                    f.write(f"{pred}\t{label}\n")
            logger.info(f"Predictions saved to {predictions_path}")

        return logits, labels

    def compute_metrics(self, eval_pred):
        """
        Compute classification metrics (accuracy, precision, recall, F1-score).
        """
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            predictions = logits[0].argmax(axis=-1)
        else:
            predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        f1_macro = f1_score(labels, predictions, average="macro")
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy, "f1-macro": f1_macro, "precision": precision, "recall": recall, "f1-weighted": f1}

    def evaluate(self, test_data, save_results=False):
        """
        Evaluate the model on test data.
        """
        logger.info("Evaluating model.")
        # trainer = Trainer(model=self.model)
        results = self.trainer.evaluate(test_data)
        if save_results:
            # Save evaluation results
            output_dir = self.config.output_dir
            results_path = os.path.join(output_dir, "evaluation_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Evaluation results saved to {results_path}")
        return results

