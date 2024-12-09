from transformers import AutoModelForCausalLM, AutoTokenizer
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
