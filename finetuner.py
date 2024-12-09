from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

class FineTuner:
    def __init__(self, model_name, num_labels, fine_tune_type, output_dir):
        """
        Initialize the FineTuner.
        Args:
            model_name (str): Hugging Face model name.
            num_labels (int): Number of classification labels.
            fine_tune_type (str): Type of fine-tuning (e.g., "lora", "adapter", "prefix").
            output_dir (str): Directory to save the model and results.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.output_dir = output_dir

        if fine_tune_type == "lora":
            self._apply_lora()

    def _apply_lora(self):
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])
        self.model = get_peft_model(self.model, config)

    def train(self, train_dataset, eval_dataset):
        """
        Fine-tune the model.
        Args:
            train_dataset: Hugging Face Dataset for training.
            eval_dataset: Hugging Face Dataset for evaluation.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            save_steps=1000,
            eval_steps=500,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
