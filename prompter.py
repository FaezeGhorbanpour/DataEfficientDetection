from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import json
import logging

logger = logging.getLogger(__name__)

class Prompter:
    def __init__(self, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Prompter with a model and tokenizer.
        Args:
            model_name (str): Hugging Face model name.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        self.device = device
        self.config = config
        self.model_name = config.prompter_model_name_or_path
        if 't5' in config.prompter_model_name_or_path:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def test(self, test_data, prompt_template, save_predictions=False):
        """
        Test the instruction-tuned model on a large dataset efficiently.
        Args:
            test_dataset (datasets.Dataset): Hugging Face dataset for testing.
            prompt_template (str): Template for prompting (e.g., "Classify: {input}").
            batch_size (int): Batch size for efficient processing.
            max_length (int): Maximum sequence length for input/output.
        Returns:
            list[str]: Model-generated responses for the dataset.
        """
        dataloader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False)
        results = []

        for batch in tqdm(dataloader, desc="Processing batches"):
            # Generate input prompts
            inputs = [prompt_template.format(input=text) for text in batch["text"]]
            tokenized_inputs = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)

            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=tokenized_inputs["input_ids"],
                    attention_mask=tokenized_inputs["attention_mask"],
                    max_length=self.config.max_length,
                    num_return_sequences=1,
                )

            # Decode responses
            decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            results.extend(decoded_outputs)

        predictions = [int(i) if isinstance(i, int) else -1 for i in results]

        if save_predictions:
            # Save predictions
            output_dir = self.config.output_dir
            predictions_path = os.path.join(output_dir, "predictions.txt")
            with open(predictions_path, "w") as f:
                f.write("Predicted\tTrue\n")
                for pred, label in zip(predictions, test_data['label']):
                    f.write(f"{pred}\t{label}\n")
            logger.info(f"Predictions saved to {predictions_path}")

        return predictions

    def form_prompt_template(self, metadata=None, k=None, ):
        if metadata is not None:
            pass
        return 'Classify the text to hate and non-hate. answer with "1" if it is hate or "0" if it non-hate. text: {input}'

    def compute_metrics(self, predictions, labels, save_results=False):
        """
        Compute classification metrics (accuracy, precision, recall, F1-score).
        """
        logger.info("Evaluating model.")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        f1_macro = f1_score(labels, predictions, average="macro")
        accuracy = accuracy_score(labels, predictions)
        results = {"accuracy": accuracy, "f1-macro": f1_macro, "precision": precision, "recall": recall, "f1-weighted": f1}

        if save_results:
            # Save evaluation results
            output_dir = self.config.prompter_output_dir
            results_path = os.path.join(output_dir, "evaluation_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Evaluation results saved to {results_path}")
        return results