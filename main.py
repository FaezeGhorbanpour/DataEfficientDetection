import argparse
from data_provider import DataProvider
from embedder import Embedder
from retriever import Retriever
from fine_tuner import FineTuner
from prompter import Prompter


def main(args):
    # Step 1: Load and prepare the data
    data_provider = DataProvider()
    datasets = data_provider.load_datasets(
        dataset_names=args.datasets,
        languages=args.languages,
        max_samples=args.max_samples,
    )

    # Step 2: Generate embeddings
    embedder = Embedder(model_name=args.embedding_model)
    embeddings, metadata = embedder.embed_datasets(datasets)

    # Step 3: Initialize and populate the retriever
    retriever = Retriever(embedding_dim=embedder.embedding_dim)
    retriever.add_embeddings(embeddings, metadata)

    # Step 4: Retrieve relevant examples for fine-tuning
    query_embedding = embedder.embed_sentences([args.query])[0].reshape(1, -1)
    retrieved_data = retriever.retrieve(
        query_embedding=query_embedding,
        k=args.k,
        filters={"language": args.query_language},
    )

    # Convert retrieved data to a format suitable for fine-tuning
    fine_tune_dataset = data_provider.convert_to_dataset(retrieved_data)

    # Step 5: Fine-tune the model
    if args.task == "fine_tune":
        fine_tuner = FineTuner(
            model_name=args.model_name,
            num_labels=args.num_labels,
            fine_tune_type=args.fine_tune_type,
            output_dir=args.output_dir,
        )
        fine_tuner.train(fine_tune_dataset, fine_tune_dataset)  # For simplicity, using the same data for val

    # Step 6: Prompt-based evaluation
    elif args.task == "prompt":
        prompter = Prompter(model_name=args.model_name)
        prompter.test(fine_tune_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Fine-Tuning Pipeline")

    # Data arguments
    parser.add_argument("--datasets", nargs="+", required=True, help="List of dataset names")
    parser.add_argument("--languages", nargs="+", required=True, help="List of languages to include")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples per dataset")

    # Embedding arguments
    parser.add_argument("--embedding_model", type=str, required=True, help="Name of the embedding model")

    # Retrieval arguments
    parser.add_argument("--query", type=str, required=True, help="Query sentence for retrieval")
    parser.add_argument("--query_language", type=str, default=None, help="Language of the query")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved examples")

    # Fine-tuning / prompting arguments
    parser.add_argument("--task", type=str, choices=["fine_tune", "prompt"], required=True, help="Task to perform")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for fine-tuning or prompting")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels")
    parser.add_argument("--fine_tune_type", type=str, default="lora", help="Fine-tuning type (lora, adapter, etc.)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models and results")

    args = parser.parse_args()
    main(args)
