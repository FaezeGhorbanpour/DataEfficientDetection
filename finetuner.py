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


class FineTunerConfig:
    """
    Configuration class for FineTuner. Used to configure the model, fine-tuning parameters,
    and training parameters.
    """

    def __init__(self, model_name, num_labels=2, fine_tune_method=None, learning_rate=5e-5, epochs=5, batch_size=16, peft_config={}):
        self.model_name = model_name
        self.num_labels = num_labels
        self.fine_tune_method = fine_tune_method  # can be "lora", "prefix_tuning", etc.
        self.learning_rate = learning_rate
        self.learning_rate = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.peft_config = peft_config  # default PEFT config (for LoRA, prefix, etc.)
        self.use_class_weight = False
        self.class_weights = None


class FineTuner:
    def __init__(self, config: FineTunerConfig):
        """
        Fine-tuning module for classification tasks with PEFT and AutoConfig.

        Args:
            config (FineTunerConfig): Configuration object with all required settings.
        """
        self.config = config
        self.model_name = config.model_name
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
        if fine_tune_method == "lora":
            config = LoraConfig(**peft_config)
        elif fine_tune_method == "prefix_tuning":
            config = PrefixTuningConfig(**peft_config)
        elif fine_tune_method == "soft_prompt":
            config = PromptEncoderConfig(**peft_config)
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
        encodings = self.tokenizer(data["text"], truncation=True, padding=True, max_length=512)
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

    def train(self, train_data, eval_data, output_dir):
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
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            overwrite_output_dir=True,
            save_total_limit=1,
            metric_for_best_model="f1-macro",

        )

        # Create Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=self.compute_metrics,
            data_collator=None,
        )

        # Use class weights in the loss function
        trainer.args.label_smoothing_factor = 0.0  # Standard loss function (use weights in the trainer)
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
        preds = logits.argmax(axis=-1)
        labels = predictions.label_ids

        return logits, labels

    def compute_metrics(self, eval_pred):
        """
        Compute classification metrics (accuracy, precision, recall, F1-score).
        """
        logits, labels = eval_pred
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
