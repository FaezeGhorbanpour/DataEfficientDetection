from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, average_precision_score, \
    recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import json
import logging
import requests

logger = logging.getLogger(__name__)

SUPPORTED_HF_MODELS = {
    "flan-t5-xxl": "google/flan-t5-xll",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-small": "google/flan-t5-small",
    "xglm": "facebook/xglm-7.5B",
    "bloomz": "bigscience/bloomz",
    "bloomz-7b1": "bigscience/bloomz-7b1",
    "mt0-base": "bigscience/mt0-base",
    "mt0-large": "bigscience/mt0-large",
    "mt0-xl": "bigscience/mt0-xl",
    "mt0-xxl": "bigscience/mt0-xxl",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B",
}
OPENROUTER_MODELS = {
    "llama-2-7b-chat": "openrouter/Meta-Llama-2-7B-Chat",
    "llama-2-13b-chat": "openrouter/Meta-Llama-2-13B-Chat",
    "llama-2-70b-chat": "openrouter/Meta-Llama-2-70B-Chat",
    "mistral-7b": "openrouter/Mistral-7B-Instruct",
    "mixtral-8x7b": "openrouter/Mixtral-8x7B-Instruct",
    "deepseek-67b": "openrouter/DeepSeek-Coder-6.7B",
    "deepseek-33b": "openrouter/DeepSeek-Coder-33B",
    "solar-10.7b": "openrouter/Solar-10.7B-Instruct",
    "yi-34b": "openrouter/Yi-34B-Chat",
    "openchat-7b": "openrouter/OpenChat-7B",
}
GROQ_MODELS = {
    "gemma-7b": "groq/gemma-7b-it",
    "gemma-2b": "groq/gemma-2b-it",
    "mixtral-8x7b-groq": "groq/mixtral-8x7b",
    "llama-2-7b-groq": "groq/llama2-7b-chat",
    "llama-2-13b-groq": "groq/llama2-13b-chat"
}
# Models accessed via OpenRouter API
OPENROUTER_API_KEY = "your_openrouter_api_key_here"  # Replace with your actual API key
GROQ_API_KEY = "your_openrouter_api_key_here"  # Replace with your actual API key


class Prompter:
    def __init__(self, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.config = config
        self.model_name = config.prompter_model_name_or_path

        if any(m in self.model_name for m in SUPPORTED_HF_MODELS):
            if 't5' in self.model_name:
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(device)
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif any(m in self.model_name for m in OPENROUTER_MODELS):
            self.use_openrouter = True
        else:
            raise NotImplementedError

    def generate_prediction(self, prompt):
        if hasattr(self, "use_openrouter"):
            return self.call_openrouter(prompt)
        else:
            tokenized_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                              max_length=self.config.prompter_max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(input_ids=tokenized_inputs["input_ids"],
                                              attention_mask=tokenized_inputs["attention_mask"],
                                              max_length=self.config.prompter_max_length)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def call_openrouter(self, prompt):
        url = "https://openrouter.ai/api/generate"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = {"model": self.model_name, "prompt": prompt, "max_tokens": 100}
        response = requests.post(url, headers=headers, json=data)
        return response.json().get("text", "")


    def call_groq(self, prompt):
        url = "https://api.groq.com/generate"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model": self.model_name, "prompt": prompt, "max_tokens": 100}
        response = requests.post(url, headers=headers, json=data)
        return response.json().get("text", "")

    def evaluate(self, test_data, retrieved_metadata):
        dataloader = DataLoader(test_data, batch_size=self.config.prompter_batch_size, shuffle=False)
        results = []

        for batch in tqdm(dataloader, desc="Processing batches"):
            for text in batch["text"]:
                retrieval_context = retrieved_metadata.get(text, "")
                prompt = self.form_prompt_template(text, retrieval_context)
                prediction = self.generate_prediction(prompt)
                results.append(int(prediction) if prediction.isdigit() and int(prediction) in [0, 1] else -1)
        return results

    def form_prompt_template(self, text, retrieved_data):
        related_texts = [entry['metadata']['text'] for entry in retrieved_data]
        prompt = f"Based on the retrieved similar texts: {'; '.join(related_texts)}, classify the following text as hate or non-hate. Answer with '1' for hate and '0' for non-hate. Text: {text}"
        return prompt

    def compute_metrics(self, predictions, labels):
        logger.info("Evaluating model.")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted",
                                                                   zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        avg_precision = average_precision_score(labels, predictions, average="macro")
        avg_recall = recall_score(labels, predictions, average="macro")

        return {"accuracy": accuracy, "f1-macro": f1_macro, "precision": precision, "recall": recall,
                "f1-weighted": f1, "average_precision": avg_precision, "average_recall": avg_recall}

    def save_results(self, predictions, labels, results, name=''):
        output_dir = self.config.prompter_output_dir
        path = os.path.join(output_dir, self.model_name.split('/')[-1], name)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "predictions.txt"), "w") as f:
            f.write("Predicted\tTrue\n")
            for pred, label in zip(predictions, labels):
                f.write(f"{pred}\t{label}\n")

        with open(os.path.join(path, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=4)







class Prompts:
    pass