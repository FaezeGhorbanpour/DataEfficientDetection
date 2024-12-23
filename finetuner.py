import torch
import logging
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
        self.fine_tune_method = config.fine_tune_method

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        unique_labels = list(set(labels))
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return torch.tensor(class_weights, dtype=torch.float)

    def train(self, train_data, eval_data):
        """
        Train the model with weighted loss and metadata considerations.
        """
        logger.info("Starting training process.")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prepare weighted loss if applicable
        if self.config.use_class_weight:
            if self.config.class_weights:
                class_weights = torch.tensor(self.config.class_weights, dtype=torch.float)
            else:
                class_weights = self.calculate_class_weights(self.config.class_weights)
            class_weights = class_weights.to(device)

        # Move weights to the appropriate device
        self.model.to(device)

        # Configure training arguments
        # training_args = TrainingArguments(
        #     output_dir=self.config.output_dir,
        #     learning_rate=self.config.learning_rate,
        #     per_device_train_batch_size=self.config.batch_size,
        #     per_device_eval_batch_size=self.config.batch_size,
        #     num_train_epochs=self.config.epochs,
        #     weight_decay=self.config.weight_decay,
        #     logging_dir=f"{self.config.output_dir}/logs",
        #     logging_steps=10,
        #     evaluation_strategy="epoch",
        #     save_strategy="epoch",
        #     overwrite_output_dir=True,
        #     save_total_limit=1,
        #     metric_for_best_model="f1-macro",
        #
        # )

        # Create Trainer instance
        trainer = Trainer(
            model=self.model,
            args=self.config,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=self.compute_metrics,
            data_collator=None,
        )

        trainer.train()

    def predict(self, test_data):
        """
        Predict labels for the test dataset.
        """
        logger.info("Running predictions.")
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_data)
        # Extract predictions and labels
        logits = predictions.predictions
        labels = predictions.label_ids

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

    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        """
        logger.info("Evaluating model.")
        trainer = Trainer(model=self.model)
        results = trainer.evaluate(test_data)
        return results
