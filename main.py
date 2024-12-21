import argparse
import logging
import wandb
from transformers import set_seed
from dataclasses import dataclass, field
from typing import Optional, List
from data_provider import DataProvider
from embedder import Embedder
from retriever import Retriever
from finetuner import FineTuner
from prompter import Prompter


# Define arguments for each module using dataclasses
@dataclass
class DataArguments:
    max_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum samples per dataset."}
    )
    languages: List[str] = field(
        default_factory=list, metadata={"help": "Languages for the datasets."}
    )
    datasets: List[str] = field(
        default_factory=list, metadata={"help": "List of dataset names to load."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to dataset cache directory."}
    )


@dataclass
class EmbedderArguments:
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Pretrained embedding model name or path."},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the embeddings."},
    )


@dataclass
class RetrieverArguments:
    top_k: int = field(
        default=10, metadata={"help": "Number of closest embeddings to retrieve."}
    )
    language_filter: Optional[List[str]] = field(
        default=None, metadata={"help": "Filter retrieved data by language."}
    )
    exclude_datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "Datasets to exclude in retrieval."}
    )
    index_type: str = field(
        default="HNSW",
        metadata={"help": "index_type"},
    )
    index_path: str = field(
        default="./index_path",
        metadata={"help": "Directory to save index path!"},
    )


@dataclass
class FineTunerArguments:
    model_name_or_path: str = field(
        default="FacebookAI/xlm-roberta-base", metadata={"help": "Pretrained model name or path."}
    )
    fine_tune_type: str = field(
        default="lora", metadata={"help": "Fine-tuning method (e.g., lora, prefix)."}
    )
    batch_size: int = field(
        default=16, metadata={"help": "Batch size for fine-tuning."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Learning rate for training."}
    )
    num_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs."}
    )
    max_length: int = field(
        default=128,
        metadata={"help": "Maximum length of the input sequence."},
    )
    fp16: bool = field(
        default=False, metadata={"help": "Enable mixed precision training (fp16)."}
    )
    logging_steps: int = field(
        default=500, metadata={"help": "Number of steps between logging."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy: 'no', 'steps', or 'epoch'."},
    )
    output_dir: str = field(
        default="./finetune_output",
        metadata={"help": "Directory to save fine-tuned models."},
    )
    do_train: bool = field(
        default=False, metadata={"help": "Set true to train the fine_tuner"}
    )
    do_test: bool = field(
        default=False, metadata={"help": "Set true to test the fine_tuner"}
    )


@dataclass
class PrompterArguments:
    model_name_or_path: str = field(
        default="google/flan-t5-base", metadata={"help": "Instruction-tuned model name or path."}
    )
    max_length: int = field(
        default=128,
        metadata={"help": "Maximum length of the input sequence for prompting."},
    )
    fp16: bool = field(
        default=False, metadata={"help": "Enable mixed precision inference (fp16)."}
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main script for multilingual NLP pipeline.")
    parser.add_argument("--wandb_project", type=str, default="multilingual_nlp_pipeline",
                        help="Wandb project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--do_embedding", action="store_true", help="Run the embedding step.")
    parser.add_argument("--do_retrieving", action="store_true", help="Run the retrieval step.")
    parser.add_argument("--do_fine_tuning", action="store_true", help="Run the fine-tuning step.")
    parser.add_argument("--do_prompter", action="store_true", help="Run the prompter step.")
    parser.add_argument("--seed", help="Random seed!")
    args, remaining_args = parser.parse_known_args()

    # Parse arguments for each module
    data_args = DataArguments(**vars(parser.parse_args(remaining_args)))
    embedder_args = EmbedderArguments(**vars(parser.parse_args(remaining_args)))
    retriever_args = RetrieverArguments(**vars(parser.parse_args(remaining_args)))
    finetuner_args = FineTunerArguments(**vars(parser.parse_args(remaining_args)))
    prompter_args = PrompterArguments(**vars(parser.parse_args(remaining_args)))

    return args, data_args, embedder_args, retriever_args, finetuner_args, prompter_args


def main():
    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Parse arguments
    args, data_args, embedder_args, retriever_args, finetuner_args, prompter_args = parse_arguments()

    # Initialize Wandb
    wandb.init(
        project=args.wandb_project,
        config=args
    )
    wandb.config.update(data_args, allow_val_change=True)
    wandb.config.update(embedder_args, allow_val_change=True)
    wandb.config.update(retriever_args, allow_val_change=True)
    wandb.config.update(finetuner_args, allow_val_change=True)
    wandb.config.update(prompter_args, allow_val_change=True)
    logger.info("Wandb initialized.")

    # Set seed before initializing model.
    set_seed(args.seed)

    # Initialize modules
    data_provider = DataProvider()

    # Step 1: Load datasets
    logger.info("Loading datasets...")
    datasets = data_provider.load_datasets(
        dataset_names=data_args.datasets,
        languages=data_args.languages,
        max_samples=data_args.max_samples
    )
    dataset = data_provider.aggregate_splits(datasets)
    logger.info("Datasets loaded: %s", [d["name"] for d in datasets])

    # Step 2: Embed datasets (optional)
    if args.do_embedding:
        embedder = Embedder(embedder_args.model_name_or_path)
        logger.info("Embedding datasets...")
        embeddings, metadatas = embedder.embed_datasets(datasets)
        logger.info("Datasets embedded with model: %s", embedder_args.model_name_or_path)



    # Step 3: Retrieve similar sentences (optional)
    retrieved_dataset = None
    if args.do_retrieving:
        if retriever_args.do_search:
            logger.info("Loading retriever's index...")
            retriever = Retriever()
            retriever.load_index(retriever_args.index_path)
            logger.info("Retrieving similar sentences...")
            retrieved_data = retriever.retrieve_multiple_queries(
                query=embeddings,
                k=retriever_args.top_k,
                filters=retriever_args.language_filter
            )
            logger.info("Retrieved %d instances based on query: %s", len(retrieved_data), query)

            # Convert retrieved data to dataset format
            retrieved_dataset = data_provider.convert_to_dataset(retrieved_data)
        else:
            retriever = Retriever(embedder.embedding_dim, index_type=retriever_args.index_type)
            retriever.add_embeddings(embeddings, metadatas)
            retriever.save_index(retriever_args.index_path)


    # Step 4: Fine-tune the model (optional)
    if args.do_fine_tuning:
        config = FineTunerConfig(
            model_name="FacebookAI/xlm-roberta-base",
            fine_tune_method="classification_head",
            num_labels=2,  # Binary classification (Hate Speech detection)
            learning_rate=5e-5,
            epochs=5,
            batch_size=16,
            # peft_config={"lora_rank": 4}
        )
        finetuner = FineTuner(config)
        logger.info("Fine-tuning the model: %s", finetuner_args.model_name_or_path)

        if finetuner_args.do_train:
            if retrieved_dataset:
                train_dataset = retrieved_dataset
                eval_dataset = _ #TODO
            else:
                train_dataset = fine_tuner.prepare_data(dataset['train'])
                eval_dataset = fine_tuner.prepare_data(dataset['validation'])

            finetuner.train(
                train_dataset, eval_dataset
            )
            logger.info("Fine-tuning completed.")
        if finetuner_args.do_test:
            test_dataset = fine_tuner.prepare_data(dataset['test'])
            predictions = fine_tuner.predict(test_dataset)
            results = fine_tuner.compute_metrics(predictions)
            results = {'fine_tuner_'+i: j for i, j in results}
            wandb.log(results)
            logger.info("Finetune-based inference metrics: %s", results)

    # Step 5: Prompt-based inference (optional)
    if args.do_prompter:
        prompter = Prompter(prompter_args.model_name_or_path)
        logger.info("Running prompt-based inference with model: %s", prompter_args.model_name_or_path)
        prompt_template = prompter.form_prompt_template()
        predictions = prompter.test(
            test_dataset=dataset[test],
            prompt_template=prompt_template
        )
        results = prompter_.compute_metrics(predictions, dataset[test]['label'])
        results = {'prompter_'+i: j for i, j in results}
        wandb.log(results)
        logger.info("Prompt-based inference metrics: %s", results)

    # Finish Wandb
    wandb.finish()
    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()
