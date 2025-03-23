import os
import json
import logging

import numpy as np
import torch
import gc
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM
)
from prompts import *
from utils import convert_to_serializable

# Configure logging
logger = logging.getLogger(__name__)

# Original Model configurations
MODEL_CONFIGS = {
    "mt0": {"name": "bigscience/mt0-large", "prompt_template": "Input: {instruction}\nOutput:", "model_type": "seq2seq",
            "context_length": 1024, "big_model": False, "batch_size": 512},
    "aya101": {"name": "CohereForAI/aya-101", "prompt_template": "Human: {instruction}\n\nnohuman:",
               "model_type": "seq2seq", "context_length": 4096, "batch_size": 32},
    "aya23": {"name": "CohereForAI/aya-23-8B", "prompt_template": "Human: {instruction}\n\nnohuman:",
              "context_length": 4096, "batch_size": 32},
    "aya8": {"name": "CohereForAI/aya-expanse-8b", "prompt_template": "Human: {instruction}\n\nnohuman:",
              "context_length": 8000, "batch_size": 32}, #new
    "bloomz": {"name": "bigscience/bloomz-7b1", "prompt_template": "{instruction}", "context_length": 2048,
               "batch_size": 32},
    "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.2", "prompt_template": "<s>[INST] {instruction} [/INST]",
                "context_length": 32768, "batch_size": 64},
    "mistral8": {"name": "mistralai/Ministral-8B-Instruct-2410", "prompt_template": "<s>[INST] {instruction} [/INST]",
                "context_length": 128000, "batch_size": 64}, #new
    "llama2": {"name": "meta-llama/Llama-2-7b-chat-hf", "prompt_template": "[INST] {instruction} [/INST]",
               "context_length": 4096, "batch_size": 64},
    "llama3": {"name": "meta-llama/Llama-3.1-8B-Instruct", "prompt_template": "[INST] {instruction} [/INST]",
               "context_length": 128000, "batch_size": 128},
    "llama3-abliterated": {"name": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", "prompt_template": "[INST] {instruction} [/INST]",
               "context_length": 128000, "batch_size": 32},
    "gemma": {"name": "google/gemma-7b-it",
              "prompt_template": "<start_of_turn>\n{instruction}<end_of_turn>\n<start_of_turn>",
              "context_length": 8192, "batch_size": 16},
    "gemma9": {"name": "google/gemma-2-9b",
              "prompt_template": "<start_of_turn>\n{instruction}<end_of_turn>\n<start_of_turn>",
              "context_length": 8192, "batch_size": 16}, #new
    "teuken": {"name": "openGPT-X/Teuken-7B-instruct-research-v0.4", "batch_size": 64,
               "prompt_template": "System: translate_to\nUser: {instruction}\nAssistant:", "context_length": 8192},
    "qwan": {"name": "Qwen/Qwen2.5-7B-Instruct", "batch_size": 128, #new
               "prompt_template": "{instruction}", "context_length": 8192}
}



class Prompter:
    """
    Enhanced BatchPrompter class for efficient processing of large-scale text data.
    """

    def __init__(self, config, device = None):
        """Initialize BatchPrompter with configuration."""
        logger.info("Initializing BatchPrompter...")

        # Basic configuration
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Model configuration
        self.model_name = config.prompter_model_name_or_path
        self.few_shot_prompts_list = (['few_shot', 'few_shot_cot', 'few_shot_role_play', 'few_shot_sole',
                                       'few_shot_multilingual', 'few_shot_definition', ]
            if config.prompts_list == 'all' else config.prompts_list
        )
        self.zero_shot_prompts_list = (['general', 'classification', 'definition', 'cot', 'multilingual', 'nli', 'sole',
                                        'role_play', 'multilingual_definition', 'role_play_cot', 'multilingual_cot',
                                        'explanation_based', 'target_identification', 'contextual_analysis', 'cot-2',
                                        'translate_then_classify', 'distinction']
            if config.prompts_list == 'all' else config.prompts_list
        )

        # Processing parameters
        self.num_rounds = config.num_rounds
        self.prompter_max_length = config.prompter_max_length
        self.model_config = self.get_model_config()

        # Load model and tokenizer
        self.tokenizer, self.model = self.load_model()
        logger.info("BatchPrompter initialization completed")

        # Memory management
        self.memory_threshold = 0.85 

    def get_model_config(self):
        """Get model configuration from MODEL_CONFIGS."""
        logger.debug(f"Getting configuration for model: {self.model_name}")
        for key in MODEL_CONFIGS:
            if key in self.model_name.lower():
                return MODEL_CONFIGS[key]
        raise ValueError(f"Model '{self.model_name}' not found in configurations")

    def load_model(self):
        """Load model and tokenizer with memory-efficient settings."""
        logger.info(f"Loading model: {self.model_config['name']}")

        try:
            # Configure quantization
            # nf4_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # )

            # Load appropriate model type
            if self.model_config.get("model_type") == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config["name"], torch_dtype=torch.float16,
                    device_map='auto' if self.model_config.get("big_model", True) else None, trust_remote_code=True,
                   ).to(device=self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_config["name"], torch_dtype=torch.float16,
                    device_map='auto' if self.model_config.get("big_model", True) else None, trust_remote_code=True,
                ).to(device=self.device)

            tokenizer = AutoTokenizer.from_pretrained(self.model_config["name"], trust_remote_code=True, 
                                                      padding_side="left")

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model.eval()
            logger.info("Model and tokenizer loaded successfully")
            return tokenizer, model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def check_memory_usage(self):
        """Monitor and manage GPU memory usage."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > self.memory_threshold:
                logger.warning("High memory usage detected. Performing cleanup...")
                torch.cuda.empty_cache()
                gc.collect()

    def batch_generate_predictions(self, prompts, max_length, lang='en'):
        """Generate predictions for a batch of prompts efficiently."""
        logger.debug(f"Generating predictions for batch of {len(prompts)} prompts")

        # Format prompts with model template
        formatted_prompts = [
            self.model_config["prompt_template"].replace('translate_to', LANGUAGES[lang]).format(instruction=prompt)
            for prompt in prompts
        ]

        # Tokenize batch
        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length).to(self.device)

        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,         # Ensure deterministic output
                num_beams=1,             # Use greedy decoding
                max_new_tokens=10,        # Allow enough tokens for "yes" or "no"
                temperature=None,         # Force deterministic output
                top_p=None,
                top_k=None,
#                min_length=inputs["input_ids"].shape[1] + 3,  # Ensure at least one token is generated
                eos_token_id=self.tokenizer.eos_token_id
            )
            # outputs = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            #                               pad_token_id=self.tokenizer.pad_token_id, do_sample=False,
            #                               num_beams=2, max_new_tokens=1, temperature=None, top_p=None, top_k=None)

        # Process predictions
        predictions = []
        template = self.model_config["prompt_template"].format(instruction="")
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            prediction = decoded.replace(template, "")
            predictions.append(prediction)

        return predictions

    def predict(self, dataset, prompt_template, max_length, batch_size, translate_prompt=False, lang='en', examples=None):
        """Process dataset and generate predictions."""
        logger.info("Starting batch prediction...")

        # Create efficient dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True) #todo this should be in dataprovider not here!

        all_predictions = []
        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Prepare batch prompts
            batch_prompts = [
                self.form_prompt(text=batch['text'][i], id=batch['id'][i], lang=lang, prompt_template=prompt_template,
                                 translate_prompt=translate_prompt,
                                 examples=examples if examples else None
                )
                for i in range(len(batch['text']))
            ]

            # Generate predictions
            batch_predictions = self.batch_generate_predictions(batch_prompts, max_length, lang=lang)

            all_predictions.extend(batch_predictions)

            # Memory management
            if (batch_idx + 1) % 50 == 0:
                self.check_memory_usage()
                logger.info(f"Processed {batch_idx + 1}/{total_batches} batches")

        return all_predictions

    def form_prompt(self, text, id, lang, prompt_template, translate_prompt=False, examples=None):
        """Form prompt using one of three general prompt categories based on input types."""
        if examples is not None:
            examples = examples[id]

        # Determine which general prompt category to use
        if 'few_shot' in prompt_template and 'multilingual' in prompt_template:
            if not examples:
                raise ValueError("Few-shot prompting requires examples.")
            prompt_method = example_and_language_based_prompt
            extra_args = {"examples": examples, "language": lang}
        elif 'few_shot' in prompt_template:
            if not examples:
                raise ValueError("Few-shot prompting requires examples.")
            prompt_method = example_based_prompt
            extra_args = {"examples": examples}
        elif 'multilingual' in prompt_template or 'translate' in prompt_template:
            if not lang:
                raise ValueError("Multilingual prompting requires language.")
            prompt_method = language_aware_prompt
            extra_args = {"language": lang}
        else:
            # Standard prompts (general, definition, classification, cot, nli, role_play)
            prompt_method = standard_prompt
            extra_args = {}

        # Handle translation
        target_lang = lang if translate_prompt else 'en'

        # Call the appropriate prompt method with all necessary args
        return prompt_method(text, variant=prompt_template, translate_to=target_lang, **extra_args)

    def abort_run(self, data, translate_prompt, prompt, split, shot_number=None, i=0):
        # Save results
        output_dir = os.path.join(self.config.prompter_output_dir, self.model_name, data['name'],
                                  data['language'] if translate_prompt else 'en', split, prompt, str(shot_number) if shot_number else '', str(i))
        file_path = os.path.join(output_dir, "evaluation_results.json")
        if os.path.exists(file_path):
            print(f"Error: The file {file_path} exist. Aborting the run.")
            return None
        return output_dir

    def do_zero_shot_prompting(self, data):
        """Perform zero-shot prompting on dataset."""
        logger.info(f"Starting zero-shot prompting for dataset: {data['name']}")

        for split in ["test", "hate_check", "hate_day"]:
            if split not in data['data']:
                continue

            dataset = data["data"][split]

            for prompt in self.zero_shot_prompts_list:
                max_length = (self.prompter_max_length or 1024 if "cot" in prompt or 'distinction' in prompt else 256)
                batch_size = self.model_config.get("batch_size", self.config.prompter_batch_size)
                batch_size = (batch_size//2 or 1024 if "cot" in prompt or 'distinction' in prompt else batch_size)
                translate_prompt = False
                for translate_prompt in [False, True]:
                    try:
                        for i in range(self.num_rounds):
                            logger.info("-" * 100)
                            logger.info(f"Starting split: {split}, prompt: {prompt}, translate_prompt: {translate_prompt}, "
                                        f"round: {i}, batch_size: {batch_size}, max_length: {max_length}")

                            output_dir = self.abort_run(data, translate_prompt=translate_prompt, prompt=prompt,
                                                        split=split, i=i)
                            if output_dir is None:
                                continue

                            predictions = self.predict(dataset, prompt, max_length=max_length, batch_size=batch_size,
                                                       translate_prompt=translate_prompt, lang=data["language"])

                            # Process predictions
                            processed_predictions = [
                                map_output(pred, lang=data['language'], translate_prompt=translate_prompt)
                                for pred in predictions
                            ]

                            results = self.compute_metrics(processed_predictions, dataset["label"])

                            self.save_predictions(dataset["id"], predictions, processed_predictions, dataset["label"], output_dir)
                            self.save_results(results, output_dir)
                    except Exception as e:
                        logger.info(f"Error in data:{data['name']} split: {split}, prompt: {prompt}!\nError: {str(e)}")
    @staticmethod
    def get_shots(shots, shot_number):
        result = {}

        for key, items in shots.items():
            label_0_items = []
            label_1_items = []

            for item in items:
                if item['label'] == 0 and len(label_0_items) < shot_number:
                    label_0_items.append(item)
                elif item['label'] == 1 and len(label_1_items) < shot_number:
                    label_1_items.append(item)

                if len(label_0_items) >= shot_number and len(label_1_items) >= shot_number:
                    break

            interleaved = []
            for i in range(min(len(label_0_items), len(label_1_items))):
                interleaved.append(label_0_items[i])
                interleaved.append(label_1_items[i])

            result[key] = interleaved

        return result

    def do_few_shot_prompting(self, data, shots):
        """Perform few-shot prompting on dataset."""
        logger.info(f"Starting few-shot prompting for dataset: {data['name']}")
        shot_numbers = [1, 3, 5, 10, 25, 50]

        for split in ["test", "hate_check"]:
            if split not in data['data']:
                continue

            dataset = data["data"][split]
            for shot_number in shot_numbers:
                examples = self.get_shots(shots, shot_number)#{key: value[:shot_number] for key, value in shots.items()}
                for prompt in self.few_shot_prompts_list:
                    max_length = (self.prompter_max_length or 512 if "cot" in prompt else 256)
                    batch_size = self.model_config.get("batch_size", self.config.prompter_batch_size)
                    # Calculate scaling factors based on shot_number
                    scaling_factor = 2
                    if shot_number >= 5:
                        scaling_factor *= 2
                    if shot_number >= 10:
                        scaling_factor *= 2
                    if shot_number >= 25:
                        scaling_factor *= 2
                    if shot_number >= 50:
                        scaling_factor *= 2

                    # Apply scaling factors
                    max_length *= scaling_factor
                    batch_size = max(batch_size // scaling_factor, 1)

                    translate_prompt = False
                    for translate_prompt in [False, True]:
                        try:
                            for i in range(self.num_rounds):
                                logger.info("-" * 100)
                                logger.info(f"Starting split: {split}, prompt: {prompt}, translate_prompt: "
                                            f"{translate_prompt}, round: {i}, shot_number:{shot_number}, batch_size: "
                                            f"{batch_size}, max length: {max_length}")

                                output_dir = self.abort_run(data, translate_prompt=translate_prompt, prompt=prompt,
                                                            split=split, shot_number=shot_number, i=i)
                                if output_dir is None:
                                    continue

                                predictions = self.predict(dataset, prompt, max_length=max_length, batch_size=batch_size, examples=examples,
                                                           translate_prompt=translate_prompt, lang=data["language"])

                                # Process predictions
                                processed_predictions = [
                                    map_output(pred, lang=data['language']) if translate_prompt else map_output(pred)
                                    for pred in predictions
                                ]

                                results = self.compute_metrics(processed_predictions, dataset["label"])

                                self.save_predictions(dataset["id"], predictions, processed_predictions, dataset["label"],
                                                      output_dir)
                                self.save_results(results, output_dir)
                        except Exception as e:
                            logger.info(f"Error in data:{data['name']} split: {split}, prompt: {prompt}!\nError: {str(e)}")

    def compute_metrics(self, predictions, labels):
        """Compute classification metrics for hate speech detection."""
        logger.info("Computing metrics...")

        # Compute precision, recall, and F1 only for class '1' (hate speech)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # Get values for class 1 (hate speech)
        hate_precision = precision[1]
        hate_recall = recall[1]
        hate_f1 = f1[1]
        hate_support = support[1]

        # Compute True Positives, False Positives, False Negatives, and True Negatives for class 1 (hate speech)
        true_positives = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
        false_positives = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
        false_negatives = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))
        true_negatives = sum((p == 0 and l == 0) for p, l in zip(predictions, labels))

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "f1-macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "f1-weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
            "precision_hate": hate_precision,
            "recall_hate": hate_recall,
            "f1_hate": hate_f1,
            "support_hate": hate_support,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }

        logger.info("Metrics computation completed: %s", metrics)
        return metrics

    def save_results(self, results, output_dir):
        """Save evaluation results."""
        try:
            results_serializable = convert_to_serializable(results)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
                json.dump(results_serializable, f, indent=4)
            logger.info(f"Results saved in %s", output_dir)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def save_predictions(self, ids, raw_predictions, predictions, labels, output_dir):
        """Save predictions and labels."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            predictions_serializable = convert_to_serializable(predictions)
            raw_predictions_serializable = convert_to_serializable(raw_predictions)
            labels_serializable = convert_to_serializable(labels)
            with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
                f.write("ID, Raw Predicted, Predicted, True\n")
                for id, raw_pred, pred, label in zip(ids, raw_predictions_serializable, predictions_serializable, labels_serializable):
                    f.write(f"{id}, {raw_pred}, {pred}, {label}\n")
            logger.info("Predictions consist of: %s", ", ".join(map(str, np.unique(predictions_serializable))))
            logger.info(f"Predictions saved. ")
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise

    def cleanup(self):
        """Perform cleanup operations."""
        logger.info("Performing cleanup...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
