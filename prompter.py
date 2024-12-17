from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

class Prompter:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Prompter with a model and tokenizer.
        Args:
            model_name (str): Hugging Face model name.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        self.device = device
        if 't5' in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def test(self, test_dataset, prompt_template, batch_size=16, max_length=512):
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
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
                    max_length=max_length,
                    num_return_sequences=1,
                )

            # Decode responses
            decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            results.extend(decoded_outputs)

        return results

    def compute_metrics(self, predictions, labels):
        """
        Compute classification metrics (accuracy, precision, recall, F1-score).
        """
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        f1_macro = f1_score(labels, predictions, average="macro")
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy, "f1-macro": f1_macro, "precision": precision, "recall": recall, "f1-weighted": f1}

