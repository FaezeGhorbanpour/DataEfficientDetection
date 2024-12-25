import argparse
import logging
import sys
import os
import wandb
import transformers
from transformers import set_seed, HfArgumentParser, TrainingArguments
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from typing import Optional, List
from data_provider import DataProvider
from embedder import Embedder
from retriever import Retriever
from finetuner import FineTuner
from prompter import Prompter

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/mounts/work/faeze/.cache/hf/'
os.environ['HF_HOME'] = '/mounts/work/faeze/.cache/hf/'
os.environ['HF_DATASETS_CACHE'] = '/mounts/work/faeze/.cache/hf/'
#os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TORCH_HUB'] = '/mounts/work/faeze/.cache/torch/'
os.environ['TORCH_HOME'] = '/mounts/work/faeze/.cache/torch/'
os.environ["WANDB_DIR"] = '/mounts/work/faeze/.cache/wandb/'
os.environ["WANDB_START_METHOD"] = 'thread'

@dataclass
class FineTunerArguments(TrainingArguments):
    """
    Arguments for FineTuner, extending HuggingFace's TrainingArguments.
    """
    finetuner_model_name_or_path: str = field(
        default="FacebookAI/xlm-roberta-base",
        metadata={"help": "Pretrained model name or path."}
    )
    finetuner_tokenizer_name_or_path: str = field(
        default="",
        metadata={"help": "Pretrained model name or path."}
    )
    fine_tune_method: str = field(
        default="default",
        metadata={"help": "Fine-tuning method (e.g., lora, prefix)."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training and evaluation."}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification tasks."}
    )
    peft_config: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for PEFT methods like LoRA or prefix tuning."}
    )
    use_class_weight: bool = field(
        default=False,
        metadata={"help": "Whether to use class weights for loss computation."}
    )
    class_weights: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Class weights for weighted loss computation."}
    )
    metric_for_best_model: str = field(
        default="f1-macro",
        metadata={"help": "Metric to use for selecting the best model during training."}
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "Number of update steps between two logs."}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of update steps between two logs."}
    )
    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "Evaluation strategy to adopt during training."}
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "Save strategy to adopt during training."}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total number of checkpoints to save."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Limit the total number of checkpoints to save."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Set true to train the FineTuner."}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Set true to train the FineTuner."}
    )
    do_test: bool = field(
        default=False,
        metadata={"help": "Set true to test the FineTuner."}
    )
    use_class_weights: Optional[str] = field(
        default=False,
        metadata = {"help": "PR custom: decide whether to use class weighting when calculating loss"}
    )


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
    sizes: List[str] = field(
        default_factory=list, metadata={"help": "List of dataset sizes to load."}
    )
    rss: List[str] = field(
        default_factory=list, metadata={"help": "List of dataset rss to load."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class EmbedderArguments:
    embedder_model_name_or_path: str = field(
        default="m3",
        metadata={"help": "Pretrained embedding model name or path."},
    )
    embedder_output_dir: Optional[str] = field(
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
    do_search: bool = field(
        default=False,
        metadata={"help": "Search."}
    )
    do_index: bool = field(
        default=False,
        metadata={"help": "Index."}
    )



@dataclass
class PrompterArguments:
    prompter_model_name_or_path: str = field(
        default="google/flan-t5-base", metadata={"help": "Instruction-tuned model name or path."}
    )
    prompter_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the embeddings."},
    )
    # max_length: int = field(
    #     default=128,
    #     metadata={"help": "Maximum length of the input sequence for prompting."},
    # )
    # fp16: bool = field(
    #     default=False, metadata={"help": "Enable mixed precision inference (fp16)."}
    # )

@dataclass
class MainArguments:
    wandb_project: str = field(
        default="DataEfficient",
        metadata={"help": "Wandb project name."}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run name."}
    )
    do_embedding: bool = field(
        default=False,
        metadata={"help": "Run the embedding step."}
    )
    do_retrieving: bool = field(
        default=False,
        metadata={"help": "Run the retrieval step."}
    )
    do_fine_tuning: bool = field(
        default=False,
        metadata={"help": "Run the fine-tuning step."}
    )
    do_prompter: bool = field(
        default=False,
        metadata={"help": "Run the prompter step."}
    )
    # seed: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "Random seed!"}
    # )



def main():
    parser = HfArgumentParser((MainArguments, DataArguments, EmbedderArguments, RetrieverArguments, FineTunerArguments, PrompterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        main_args, data_args, embedder_args, retriever_args, finetuner_args, prompter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        main_args, data_args, embedder_args, retriever_args, finetuner_args, prompter_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if is_main_process(finetuner_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {finetuner_args.local_rank}, device: {finetuner_args.device}, n_gpu: {finetuner_args.n_gpu}"
        + f"distributed training: {bool(finetuner_args.local_rank != -1)}, 16-bits training: {finetuner_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(finetuner_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {finetuner_args}")

    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # Parse arguments
    # main_args, data_args, embedder_args, retriever_args, finetuner_args, prompter_args = parse_arguments()

    # Initialize Wandb
    wandb.init(
        project=main_args.wandb_project,
        name=f"{main_args.wandb_run_name}-{data_args.datasets[0]}-{finetuner_args.seed}",
        config=main_args,
    )
    wandb.config["log_frequency"] = 1000
    wandb.config["log_model"] = False
    logger.info("Wandb initialized.")

    # Set seed before initializing model.
    set_seed(finetuner_args.seed)

    # Initialize modules
    data_provider = DataProvider()
    data_args.sizes = [x.split('-')[1] if '-' in data_args.datasets else 'full'for x in data_args.datasets]
    data_args.rss = [x.split('-')[2] if '-' in data_args.datasets else 'full' for x in data_args.datasets]
    wandb.config.update(data_args, allow_val_change=False)

    # Step 1: Load datasets
    logger.info(f"Loading datasets: {data_args.datasets} ...")
    datasets = data_provider.load_datasets(
        dataset_names=data_args.datasets,
        languages=data_args.languages,
        cache_dir=data_args.dataset_cache_dir,
        max_samples=data_args.max_samples
    )
    logger.info("Datasets loaded: %s", [d["name"] for d in datasets])

    # Step 2: Embed datasets (optional)
    if main_args.do_embedding:
        embedder = Embedder(embedder_args.embedder_model_name_or_path)
        wandb.config.update(embedder_args, allow_val_change=False)
        logger.info("Embedding datasets...")
        embeddings, metadatas = embedder.embed_datasets(datasets)
        logger.info("Datasets embedded with model: %s", embedder_args.embedder_model_name_or_path)



    # Step 3: Retrieve similar sentences (optional)
    retrieved_dataset = None
    if main_args.do_retrieving:
        wandb.config.update(retriever_args, allow_val_change=False)
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
    else:
        dataset = data_provider.aggregate_splits([dataset['data'] for dataset in datasets])



    # Step 4: Fine-tune the model (optional)
    if main_args.do_fine_tuning:
        wandb.config.update(finetuner_args, allow_val_change=False)
        fine_tuner = FineTuner(finetuner_args)
        logger.info("Fine-tuning the model: %s", finetuner_args.finetuner_model_name_or_path)

        if finetuner_args.do_train:
            if retrieved_dataset:
                train_dataset = retrieved_dataset
                eval_dataset = _ #TODO
            else:
                train_dataset = fine_tuner.prepare_data(dataset['train'])
                eval_dataset = fine_tuner.prepare_data(dataset['validation'])

            fine_tuner.train(
                train_dataset, eval_dataset
            )
            logger.info("Fine-tuning completed.")
        if finetuner_args.do_test:
            test_dataset = fine_tuner.prepare_data(dataset['test'])
            predictions = fine_tuner.predict(test_dataset, True)
            results = fine_tuner.evaluate(test_dataset, True)
            # results = {'fine_tuner_'+i: j for i, j in results.items()}
            wandb.log(results)
            logger.info("Finetune-based inference metrics: %s", results)

    # Step 5: Prompt-based inference (optional)
    if main_args.do_prompter:
        wandb.config.update(prompter_args, allow_val_change=False)
        prompter = Prompter(prompter_args.prompter_model_name_or_path)
        logger.info("Running prompt-based inference with model: %s", prompter_args.prompter_model_name_or_path)
        prompt_template = prompter.form_prompt_template()
        predictions = prompter.test(
            test_dataset=dataset[test],
            prompt_template=prompt_template
        )
        results = prompter_.compute_metrics(predictions, dataset[test]['label'])
        # results = {'prompter_'+i: j for i, j in results.items()}
        wandb.log(results)
        logger.info("Prompt-based inference metrics: %s", results)

    # Finish Wandb
    wandb.finish()
    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()
