import evaluate


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
        return self.perplexity_metric.compute(predictions=sentences, model_id=self.model_name,
                                              batch_size=self.batch_size, device=self.device)['perplexities']

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
