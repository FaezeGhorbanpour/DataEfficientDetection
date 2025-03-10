import evaluate


import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class UncertaintyCalculator:
    def __init__(self, model_name, tokenizer_model_name, batch_size=32, device: str = 'cuda'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.batch_size = batch_size
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def calculate_uncertainty_batch(self, texts):
        """Computes entropy-based uncertainty for a batch of texts."""
        entropies = []
        margins = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(self.device)

            with torch.no_grad():  # **Disable gradient computation for efficiency**
                outputs = self.model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            batch_entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)  # Entropy calculation


            # Margin-based uncertainty
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # Difference between top two classes

            entropies.extend([float(i) for i in batch_entropies])
            margins.extend([float(i) for i in margin])

        return entropies, margins



class PerplexityCalculator:
    def __init__(self, model_name: str, batch_size: int = 32, device: str = 'cuda'):
        """
        Initialize the PerplexityCalculator with the specified model.
        Args:
            model_name (str): Hugging Face model name (e.g., 'gpt2').
            batch_size (int): Batch size for processing sentences efficiently.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Load perplexity metric from the 'evaluate' library
        self.perplexity_metric = evaluate.load("perplexity", module_type="metric")

    def calculate_perplexity_batch(self, sentences):
        """
        Calculate perplexity for a batch of sentences using the 'evaluate' library.
        Args:
            sentences (List[str]): List of sentences for which to calculate perplexity.
        Returns:
            List[float]: List of perplexities corresponding to each sentence.
        """
        # Filter out empty strings or strings that might tokenize to empty sequences
        valid_sentences = []
        invalid_indices = []

        for i, sentence in enumerate(sentences):
            # Skip completely empty strings or whitespace-only strings
            if not sentence or sentence.isspace():
                invalid_indices.append(i)
            else:
                valid_sentences.append(sentence)

        if not valid_sentences:
            return [float('inf')] * len(sentences)  # Return infinity for all if no valid sentences

        # Calculate perplexity only for valid sentences
        valid_perplexities = self.perplexity_metric.compute(
            predictions=valid_sentences,
            model_id=self.model_name,
            batch_size=self.batch_size,
            device=self.device
        )['perplexities']

        # Reconstruct the full results with placeholders for invalid sentences
        result = []
        valid_idx = 0

        for i in range(len(sentences)):
            if i in invalid_indices:
                result.append(float('inf'))  # Use infinity for invalid sentences
            else:
                result.append(valid_perplexities[valid_idx])
                valid_idx += 1

        return result

    def rank_sentences_by_perplexity(self, sentences):
        """
        Rank sentences by their perplexity (higher perplexity means more unusual).
        Args:
            sentences (List[str]): List of sentences to rank.
        Returns:
            List[str]: Sentences ranked by perplexity in descending order.
        """
        # Calculate perplexities for all sentences
        perplexities =  self.calculate_perplexity_batch(sentences)

        # Rank sentences based on perplexity
        ranked_sentences = sorted(zip(sentences, perplexities), key=lambda x: x[1], reverse=True)

        return [sentence for sentence, _ in ranked_sentences]


def z_score_normalizer(input):
    input = np.array(input)
    # Z-score Normalization
    mean = input.mean()
    std = input.std()
    z_normalized_uncertainties = (input - mean) / (std + 1e-10)

    return z_normalized_uncertainties

def minmax_normalizer(input):
    input = np.array(input)

    # Min-Max Normalization
    min_val = input.min()
    max_val = input.max()
    minmax_normalized_uncertainties = (input - min_val) / (max_val - min_val + 1e-10)

    return minmax_normalized_uncertainties


# Example usage:
if __name__ == "__main__":
    model_name = "facebook/xglm-564M"  # You can change this to any model you'd like to use
    calculator = PerplexityCalculator(model_name=model_name, batch_size=16, device='cuda')

    # Example multilingual sentences
    sentences = [
        "Das ist ein Beispiel für einen Satz.",  # German
        "Este es un ejemplo de oración.",  # Spanish
        "यह एक उदाहरण वाक्य है।",  # Hindi
        "هذا مثال على جملة.",  # Arabic
        "Odio a ciertas personas por su raza.",  # Hate speech (Spanish)
    ]

    print(calculator.calculate_perplexity_batch(sentences))

    # Rank sentences by perplexity
    ranked_sentences = calculator.rank_sentences_by_perplexity(sentences)

    print("Top high-perplexity samples for fine-tuning:")
    for sentence in ranked_sentences:
        print(sentence)


