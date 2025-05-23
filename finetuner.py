import abc
import shutil

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
    DataCollatorWithPadding, EarlyStoppingCallback,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, PrefixTuningConfig, PromptEncoderConfig, TaskType
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from trainers import WeightedTrainer, RetrievalWeightedTrainer, CurriculumLearningTrainer

logger = logging.getLogger(__name__)

instruction_translations = {
    "en": "Is this comment hateful?",
    "es": "¿Es este comentario de odio?",
    "hi": "क्या यह टिप्पणी घृणास्पद है?",
    "it": "Questo commento è offensivo?",
    "de": "Ist dieser Kommentar hasserfüllt?",
    "ar": "هل هذا التعليق يحض على الكراهية؟",
    "tr": "Bu yorum nefret dolu mu?",
    "pt": "Este comentário é odioso?",
    "fr": "Ce commentaire est-il haineux?",
}

label_translations = {
    "en": {0: "no", 1: "yes"},
    "es": {0: "no", 1: "sí"},
    "hi": {0: "नहीं", 1: "हाँ"},
    "it": {0: "no", 1: "sì"},
    "de": {0: "nein", 1: "ja"},
    "ar": {0: "لا", 1: "نعم"},
    "tr": {0: "hayır", 1: "evet"},
    "pt": {0: "não", 1: "sim"},
    "fr": {0: "non", 1: "oui"},
}

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
        self.retrieval_loss_weight = getattr(config, 'retrieval_loss_weight', None)
        self.do_early_stopping = getattr(config, 'do_early_stopping', None)
        self.use_step_trainer = getattr(config, 'use_step_trainer', None)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Configure and load the model
        model_config = AutoConfig.from_pretrained(self.model_name, num_labels=config.num_labels)
        if 't5' in self.model_name or 't0' in self.model_name:
            self.config.model_type = "encoder_decoder"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, config=model_config)
        else:
            self.config.model_type = "encoder"
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=model_config)


        logger.info("Starting training process.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move weights to the appropriate device
        self.model.to(self.device)

        self.eval_epoch = 0
        self.save_more = config.save_more


        # Apply PEFT if specified
        if self.fine_tune_method in ["lora", "prefix_tuning", "soft_prompt", "compactor"]:
            logger.info("Directing to peft config to be applied.")
            self._apply_peft(self.fine_tune_method, config.peft_config)

    def _apply_peft(self, fine_tune_method, peft_config):
        """
        Apply Parameter-Efficient Fine-Tuning (PEFT) to the model.
        """
        logger.info(f"Applying {fine_tune_method} fine-tuning.")
        task_type = TaskType.SEQ_2_SEQ_LM
        peft_config['task_type'] = task_type
        if fine_tune_method == "lora":
            logger.info("PEFT method: LoRA")
            config = LoraConfig(task_type=task_type, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        elif fine_tune_method == "prefix_tuning":
            logger.info("PEFT method: Prefix tuning")
            config = PrefixTuningConfig(task_type=task_type, num_virtual_tokens=10)
        elif fine_tune_method == "soft_prompt":
            logger.info("PEFT method: Soft prompt tuning")
            config = PromptEncoderConfig(task_type=task_type, num_virtual_tokens=10)
        else:
            logger.info(f"Unsupported PEFT method: {fine_tune_method}, facing error and exiting. Please check the config file and try again.")
            raise ValueError(f"Unsupported PEFT method: {fine_tune_method}")
        self.model = get_peft_model(self.model, config)

        logger.info("PEFT model initialized.")


    def prepare_data(self, data):
        if self.config.model_type in ["encoder_decoder", "causal_lm"]:
            logger.info("Preparing data for encoder-decoder or causal language model.")
            languages = data["language"]  # Directly use existing languages (no detection)

            instructions = [instruction_translations.get(lang, instruction_translations["en"]) for lang in languages]
            formatted_inputs = [f"{instr} {text}" for instr, text in zip(instructions, data["text"])]
            labels_text = [label_translations.get(lang, label_translations["en"])[label] for lang, label in
                           zip(languages, data["label"])]

            inputs = self.tokenizer(
                formatted_inputs,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length
            )

            targets = self.tokenizer(
                labels_text,
                truncation=True,
                padding="max_length",
                max_length=5
            )

            # Correctly mask padding tokens
            input_ids = torch.tensor(targets["input_ids"])
            input_ids[input_ids == self.tokenizer.pad_token_id] = -100
            targets["input_ids"] = input_ids.tolist()

            dataset = Dataset.from_dict({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": targets["input_ids"],
            })
            dataset = dataset.add_column("language", languages)

        else:
            logger.info("Preparing data for encoder-only model.")
            encodings = self.tokenizer(
                data["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length
            )

            dataset = Dataset.from_dict({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": data["label"],
            })
            dataset = dataset.add_column("language", data["language"])

        if not self.config.remove_unused_columns and 'source' in data.features:
            dataset = dataset.add_column("source", data["source"])
        if not self.config.remove_unused_columns and 'score' in data.features:
            dataset = dataset.add_column("score", data["score"])

        logger.info("Data preparation completed.")
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

        if self.config.use_class_weights:

            logger.info("** USING WEIGHTED LOSS**")
            if len(pd.unique(train_data["label"])) < 2:  # if a small sample only has examples of one class, we cannot do weighting
                class_weights = self.calculate_class_weights(eval_data["label"])
            else:
                class_weights = self.calculate_class_weights(train_data["label"])

            logger.info(f"class weights: {class_weights}")
            class_weights = class_weights.to(self.device)

            self.trainer = WeightedTrainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                class_weights=class_weights
            )
        elif self.retrieval_loss_weight and self.retrieval_loss_weight < 1:

            logger.info("** USING RETRIEVAL-WEIGHTED LOSS **")

            data_collator = RetrievalWeightedDataCollator(tokenizer=self.tokenizer)
            self.trainer = RetrievalWeightedTrainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                data_collator=data_collator,
                retrieval_loss_weight=self.retrieval_loss_weight,
            )
        elif self.config.use_curriculum_learning:
            logger.info("** USING CURRICULUM LEARNING **")

            data_collator = RetrievalWeightedDataCollator(tokenizer=self.tokenizer)

            self.trainer = CurriculumLearningTrainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                data_collator=data_collator,
                total_epochs=self.config.num_train_epochs,
                schedule_type=self.config.curriculum_schedule,
                schedule_order=self.config.curriculum_order
            )
        elif self.use_step_trainer:
            logger.info("** USING STEP-TRAIN **")
            self.config.max_steps=2000
            # self.config.warmup_steps=int(0.1 * 2000)  # 10% warmup
            self.config.evaluation_strategy="steps"  # Evaluate periodically
            self.config.eval_steps=250  # Evaluate every 250 steps
            self.config.save_steps=500  # Save checkpoints every 500 steps
            self.config.logging_steps=250  # Log progress every 50 steps
            # self.config.learning_rate=2e-5
            # self.config.weight_decay=0.01
            # self.config.max_seq_length=256
            self.config.load_best_model_at_end=True  # Ensure best checkpoint is used
            self.config.metric_for_best_model="f1-macro"  # Track F1-macro for early stopping
            self.config.greater_is_better=True  # Higher F1-macro is better
            self.report_to = ['wandb']


            self.trainer = Trainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if self.do_early_stopping else None,
            )
        elif self.config.model_type in ["encoder_decoder", "causal_lm"]:

            logger.info("** USING SEQ2SEQ Trainer**")

            seq2seq_valid_keys = set(Seq2SeqTrainingArguments.__dataclass_fields__.keys())
            filtered_config = {k: v for k, v in self.config.to_dict().items() if k in seq2seq_valid_keys}
            config = Seq2SeqTrainingArguments(**filtered_config)

            config.predict_with_generate = True
            config.generation_max_length = 5
            config.generation_num_beams = 1

            self.trainer = Seq2SeqTrainer(
                model=self.model,
                args=config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=10)] if self.do_early_stopping else None,
            )


        else:

            logger.info("** USING STANDARD UNWEIGHTED LOSS**")

            self.trainer = Trainer(
                model=self.model,
                args=self.config,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=self.compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] if self.do_early_stopping else None,
            )

        self.trainer.train()

        if self.save_more and self.config.use_curriculum_learning:
            self.save(self.trainer.epoch_train_size, "epoch_train_size.json")


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


        if self.config.model_type in ["encoder_decoder", "causal_lm"]:
            preds, labels = eval_pred

            post_processor = PostProcessor(self.tokenizer, True)
            decoded_preds, decoded_labels = post_processor.process(preds, labels,)

            # Map decoded text to binary labels
            preds_binary = [
                1 if p.strip().lower() in ["yes", "hate", "sí", "sì", "نعم", "ja", "evet", "sim", "हाँ"] else 0 for p in
                decoded_preds]
            labels_binary = [
                1 if l.strip().lower() in ["yes", "hate", "sí", "sì", "نعم", "ja", "evet", "sim", "हाँ"] else 0 for l in
                decoded_labels]

        else:
            logits, labels = eval_pred
            if isinstance(logits, tuple):
                predictions = logits[0].argmax(axis=-1)
            else:
                predictions = logits.argmax(axis=-1)
            preds_binary = predictions
            labels_binary = labels

        precision, recall, f1, _ = precision_recall_fscore_support(labels_binary, preds_binary, average="weighted", zero_division=0)
        f1_macro = f1_score(labels_binary, preds_binary, average="macro", zero_division=0)
        accuracy = accuracy_score(labels_binary, preds_binary)
        results = {"accuracy": accuracy, "f1-macro": f1_macro, "precision": precision, "recall": recall, "f1-weighted": f1}

        if self.save_more:
            self.eval_epoch += 1
            self.save(results, f"eval_{self.eval_epoch}_results.json")

        return results

    def save(self, data, file_name):
            # Save evaluation results
        output_dir = self.config.output_dir
        results_path = os.path.join(output_dir, file_name)
        with open(results_path, "w") as f:
            json.dump(data, f, indent=4)

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
            self.save(results,  f"{key}_results.json")

        return results

    def save_model(self):
        import os
        import shutil
        from distutils.dir_util import copy_tree

        output_dir = self.config.output_dir
        best_model_path = self.trainer.state.best_model_checkpoint

        if best_model_path is None:
            raise ValueError("No best model checkpoint found. Make sure `load_best_model_at_end=True` was set.")

        save_path = os.path.join(output_dir, "the_best_checkpoint")

        # Make sure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Copy all files from the best checkpoint directory, overwriting any existing files
        copy_tree(best_model_path, save_path)

        # Verify that the expected model files exist after copying
        expected_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5",
                          "model.ckpt.index", "flax_model.msgpack"]

        found_any = False
        for file in expected_files:
            if os.path.exists(os.path.join(save_path, file)):
                found_any = True
                print(f"Successfully saved {file}")

        if not found_any:
            print("Warning: None of the expected model files were found in the source directory.")
            print(f"Source directory contents: {os.listdir(best_model_path)}")

        return save_path


class RetrievalWeightedDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Extract 'source' and 'score' values for each instance
        sources = [f.pop("source") for f in features]  # Keeps per-instance source
        scores = [f.pop("score") for f in features]  # Keeps per-instance score
        languages = [f.pop("language") for f in features]  # Keeps per-instance score

        # Tokenize normally using the parent collator
        batch = super().__call__(features)

        # Convert 'source' to a tensor (1 for retrieved, 0 for main)
        batch["source"] = torch.tensor(sources, dtype=torch.int)

        # Convert 'score' to a tensor (keep original float values)
        batch["score"] = torch.tensor(scores, dtype=torch.float)

        return batch

class PostProcessor(abc.ABC):
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""
    def __init__(self, tokenizer, ignore_pad_token_for_loss):
       self.tokenizer = tokenizer
       self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def process(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels
