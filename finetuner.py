import torch
import logging
import os, json
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    DataCollator, DefaultDataCollator, DataCollatorWithPadding,
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
        self.tokenizer_name = config.finetuner_tokenizer_name_or_path if config.finetuner_tokenizer_name_or_path != '' else self.model_name
        self.fine_tune_method = config.fine_tune_method
        self.shuffle = config.shuffle
        self.retrieval_loss_weight = config.retrieval_loss_weight

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Configure and load the model
        model_config = AutoConfig.from_pretrained(self.model_name, num_labels=config.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=model_config)


        logger.info("Starting training process.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move weights to the appropriate device
        self.model.to(self.device)


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
        if 'source' in data.features:
            dataset = dataset.add_column("source", data["source"])
        if 'score' in data.features:
            dataset = dataset.add_column("score", data["score"])
        return dataset

    def calculate_class_weights(self, labels):
        """
        Calculate class weights for imbalanced datasets.
        """
        unique_labels = np.array(list(set(labels)))
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return torch.tensor(class_weights, dtype=torch.float)

    def train(self, train_dataset, eval_dataset):
        """
        Train the model with weighted loss and metadata considerations.
        """
        train_data = self.prepare_data(train_dataset)
        eval_data = self.prepare_data(eval_dataset)

        if self.shuffle:
            train_data = train_data.shuffle(seed=42)

        if self.config.use_class_weights:

            logger.info("** USING WEIGHTED LOSS**")
            if len(pd.unique(train_data["label"])) < 2:  # if a small sample only has examples of one class, we cannot do weighting
                class_weights = self.calculate_class_weights(eval_data["label"])
            else:
                class_weights = self.calculate_class_weights(train_data["label"])

            logger.info(f"class weights: {class_weights}")
            class_weights = class_weights.to(self.device)

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
        elif self.retrieval_loss_weight < 1:

            logger.info("** USING RETRIEVAL-WEIGHTED LOSS **")
            retrieval_loss_weight = self.retrieval_loss_weight
            class RetrievalWeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    source = inputs.pop("source")  # Extract "source" feature
                    score = inputs.pop("score")  # Extract "source" feature

                    outputs = model(**inputs)
                    logits = outputs.get("logits")

                    # Standard loss function (no reduction to compute separately)
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    losses = loss_fct(logits, labels)

                    # Create masks for "main" and "retrieved" samples
                    retrieval_mask = (source == 1).float().to(losses.device)  # 1 for retrieved, 0 for main
                    main_mask = 1 - retrieval_mask  # 1 for main, 0 for retrieved

                    # Compute separate losses
                    loss_main = (losses * main_mask).sum() / main_mask.sum().clamp(min=1)  # Avoid division by zero
                    loss_retrieved = (losses * retrieval_mask).sum() / retrieval_mask.sum().clamp(min=1)

                    # Final weighted loss
                    final_loss = loss_main + loss_retrieved * retrieval_loss_weight

                    return (final_loss, outputs) if return_outputs else final_loss

            class RetrievalWeightedDataCollator(DataCollatorWithPadding):
                def __call__(self, features):
                    # Extract 'source' and 'score' values for each instance
                    sources = [f.pop("source") for f in features]  # Keeps per-instance source
                    scores = [f.pop("score") for f in features]  # Keeps per-instance score

                    # Tokenize normally using the parent collator
                    batch = super().__call__(features)

                    # Convert 'source' to a tensor (1 for retrieved, 0 for main)
                    batch["source"] = torch.tensor([1 if s == "retrieved" else 0 for s in sources], dtype=torch.float)

                    # Convert 'score' to a tensor (keep original float values)
                    batch["score"] = torch.tensor(scores, dtype=torch.float)

                    return batch

            data_collator = RetrievalWeightedDataCollator(tokenizer=self.tokenizer)
            self.trainer = RetrievalWeightedTrainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                data_collator=data_collator
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

        self.trainer.train()

    def predict(self, test_dataset, save_prediction=False):
        """
        Predict labels for the test dataset.
        """
        logger.info("Running predictions.")
        # trainer = Trainer(model=self.model)
        test_data = self.prepare_data(test_dataset)
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
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy, "f1-macro": f1_macro, "precision": precision, "recall": recall, "f1-weighted": f1}

    def evaluate(self, test_dataset, save_results=False, key='evaluation', metric_key_prefix='test'):
        """
        Evaluate the model on test data.
        """
        logger.info("Evaluating model.")
        # trainer = Trainer(model=self.model)
        test_data = self.prepare_data(test_dataset)
        results = self.trainer.evaluate(test_data, metric_key_prefix=metric_key_prefix)
        if save_results:
            # Save evaluation results
            output_dir = self.config.output_dir
            results_path = os.path.join(output_dir, f"{key}_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Evaluation results saved to {results_path}")
        return results

    def save_model(self,):
        output_dir = self.config.output_dir
        model_path = os.path.join(output_dir, "checkpoint-retrieval-finetuner")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        return model_path

