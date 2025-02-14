import argparse
import sys
import logging
import os
import gc
import copy
import wandb
import torch
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
os.environ['TRANSFORMERS_CACHE'] = '/mounts/data/proj/faeze/.cache/hf/'
os.environ['HF_HOME'] = '/mounts/data/proj/faeze/.cache/hf/'
os.environ['HF_DATASETS_CACHE'] = '/mounts/data/proj/faeze/.cache/hf/'
#os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TORCH_HUB'] = '/mounts/data/proj/faeze/.cache/torch/'
os.environ['TORCH_HOME'] = '/mounts/data/proj/faeze/.cache/torch/'
os.environ["WANDB_DIR"] = '/mounts/data/proj/faeze/.cache/wandb/'
os.environ["WANDB_START_METHOD"] = 'thread'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

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
    dataloader_num_workers: int = field(
        default=2,
        metadata={"help": "Batch size for training and evaluation."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training and evaluation."}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training and evaluation."}
    )
    per_device_eval_batch_size: int = field(
        default=64,
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
    do_hate_check: bool = field(
        default=False,
        metadata={"help": "Run the hate check evaluation."}
    )
    use_class_weights: Optional[str] = field(
        default=False,
        metadata = {"help": "PR custom: decide whether to use class weighting when calculating loss"}
    )
    report_to: List[str] = field(
         default_factory=list, metadata={"help": "List of dataset sizes to load."}
    )
    repeat_target_train_set: int = field(
        default=1,
        metadata={"help": "Repeat target train set."}
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Remove unused columns."}
    )
    retrieval_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Retrieval loss weight."}
    )
    use_curriculum_learning: bool=field(
        default=False,
        metadata={"help": "Use curriculum learning for finetuning."}
    )
    curriculum_schedule: str=field(
        default='linear',
        metadata={"help": "curriculum learning schedule: linear, exponential, stepwise"}
    )
    curriculum_order: str=field(
        default='ascending',
        metadata={"help": "curriculum learning order: ascending, descending, strict_descending"}
    )
    save_more: bool=field(
        default=False,
        metadata={"help": "Set true to save more eval results."}
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
    splits: Optional[List[str]] = field(
        default=None, metadata={"help": "What splits of given datasets must be embedded."}
    )
    add_perplexity: bool = field(
        default=False,
        metadata={"help": "Add perplexity."}
    )
    add_uncertainty: bool = field(
        default=False,
        metadata={"help": "Add uncertainty."}
    )



@dataclass
class RetrieverArguments:
    k: int = field(
        default=0, metadata={"help": "Number of closest embeddings to retrieve."}
    )
    exclude_languages: Optional[List[str]] = field(
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
    max_retrieved: int = field(
        default=20000,
        metadata={"help": "Maximum retrieved instances from the pool."},
    )
    normalize_index: bool = field(
        default=False,
        metadata={"help": "Normalize the saved and retrieved embeddings."},
    )
    cluster_criteria_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for diversity criteria to score retrieved embeddings."},
    )
    unique_word_criteria_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for number of unique words criteria to score retrieved embeddings."},
    )
    balance_labels: bool = field(
        default=False,
        metadata={"help": "Balance the retrieved embeddings."},
    )



@dataclass
class RetrievalTunerArguments:
    retrieval_fine_tune_method: str = field(
        default="default",
        metadata={"help": "Fine-tuning method (e.g., lora, prefix)."}
    )
    retrieval_num_train_epochs: int = field(
        default=5,
        metadata={"help": "Batch size for training and evaluation."}
    )
    retrieval_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training and evaluation."}
    )
    retrieval_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for training and evaluation."}
    )
    retrieval_num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification tasks."}
    )
    retrieval_peft_config: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for PEFT methods like LoRA or prefix tuning."}
    )
    retrieval_max_seq_length: int = field(
        default=128,
        metadata={"help": "Limit the total number of checkpoints to save."}
    )
    retrieval_do_train: bool = field(
        default=False,
        metadata={"help": "Set true to train the FineTuner."}
    )
    retrieval_do_eval: bool = field(
        default=False,
        metadata={"help": "Set true to train the FineTuner."}
    )
    retrieval_do_test: bool = field(
        default=False,
        metadata={"help": "Set true to test the FineTuner."}
    )
    combine_train_set: bool = field(
        default=False,
        metadata={"help": "Combine retrieved data with training set."}
    )
    random_retrieve: bool = field(
        default=False,
        metadata={"help": "Combine retrieved data with training set."}
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
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    prompter_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for training and evaluation."}
    )
    prompter_max_length: int = field(
        default=128,
        metadata={"help": "Batch size for training and evaluation."}
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
    do_searching: bool = field(
        default=False,
        metadata={"help": "Search."}
    )
    do_indexing: bool = field(
        default=False,
        metadata={"help": "Index."}
    )
    do_retrieval_tuning: bool = field(
        default=False,
        metadata={"help": "Run the fine-tuning step on retrieved instances."}
    )
    do_fine_tuning: bool = field(
        default=False,
        metadata={"help": "Run the fine-tuning step."}
    )
    do_prompting: bool = field(
        default=False,
        metadata={"help": "Run the prompter step."}
    )
    enable_wandb: bool = field(
        default=True,
        metadata={"help": "Run the prompter step."}
    )
    # seed: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "Random seed!"}
    # )

def copy_args(retrieval_tuner_args, finetuner_args):
    finetuner_args_copy = copy.deepcopy(finetuner_args)
    finetuner_args_copy.retrieval_fine_tune_method = retrieval_tuner_args.retrieval_fine_tune_method
    finetuner_args_copy.num_train_epochs = retrieval_tuner_args.retrieval_num_train_epochs
    finetuner_args_copy.per_device_train_batch_size = retrieval_tuner_args.retrieval_train_batch_size
    finetuner_args_copy.per_device_eval_batch_size = retrieval_tuner_args.retrieval_eval_batch_size
    finetuner_args_copy.num_labels = retrieval_tuner_args.retrieval_num_labels
    finetuner_args_copy.peft_config = retrieval_tuner_args.retrieval_peft_config
    finetuner_args_copy.max_seq_length = retrieval_tuner_args.retrieval_max_seq_length
    finetuner_args_copy.do_train = retrieval_tuner_args.retrieval_do_train
    finetuner_args_copy.do_eval = retrieval_tuner_args.retrieval_do_eval
    finetuner_args_copy.do_test = retrieval_tuner_args.retrieval_do_test
    finetuner_args_copy.report_to = []
    return finetuner_args_copy



def main():
    parser = HfArgumentParser((MainArguments, DataArguments, EmbedderArguments, RetrieverArguments, RetrievalTunerArguments, FineTunerArguments, PrompterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        main_args, data_args, embedder_args, retriever_args, retrieval_tuner_args, finetuner_args, prompter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        main_args, data_args, embedder_args, retriever_args, retrieval_tuner_args, finetuner_args, prompter_args = parser.parse_args_into_dataclasses()

    if main_args.do_fine_tuning:
        file_path = os.path.join(finetuner_args.output_dir, "evaluation_results.json")
        if os.path.exists(file_path):
            print(f"Error: The file {file_path} exist. Aborting the run.")
            sys.exit(1)  # Abort the run

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG if needed
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
    #print(main_args.wandb_project)
    #print(f"{main_args.wandb_run_name}-{data_args.datasets[0]}-{finetuner_args.seed}")
    # Initialize Wandb
    if main_args.enable_wandb:
        wandb.init(
            project=main_args.wandb_project,
            name=f"{main_args.wandb_run_name}-{data_args.datasets[0]}-{finetuner_args.seed}",
            config=main_args,
        )
        # wandb.config["log_frequency"] = 1000
        # wandb.config["log_model"] = False
        logger.info("Wandb initialized.")

    if finetuner_args.report_to is None or finetuner_args.report_to == 'None' or 'None' in finetuner_args.report_to:
        finetuner_args.report_to = []
    elif finetuner_args.report_to == 'wandb':
        finetuner_args.report_to = ['wandb']
    
    print('excluded_datasets', retriever_args.exclude_datasets)
    # Set seed before initializing model.
    set_seed(finetuner_args.seed)

    # Initialize modules
    data_provider = DataProvider()
    data_args.sizes = [x.split('-')[1] if '-' in x else 'full'for x in data_args.datasets]
    data_args.rss = [x.split('-')[2] if '-' in x else 'full' for x in data_args.datasets]
    if main_args.enable_wandb:
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

    # Step 2: Embed datasets
    embedder = None
    embeddings, meta_datas = None, []
    if main_args.do_embedding:
        embedder = Embedder(embedder_args.embedder_model_name_or_path, add_perplexity=embedder_args.add_perplexity, add_uncertainty=embedder_args.add_uncertainty)
        if main_args.enable_wandb:
            wandb.config.update(embedder_args, allow_val_change=False)
        logger.info("Embedding datasets...")

        embeddings, meta_datas = embedder.embed_datasets(datasets, splits=embedder_args.splits)
        logger.info("Datasets embedded with model: %s", embedder_args.embedder_model_name_or_path)




    # Step 3: Retrieve similar sentences
    if main_args.do_indexing:
        if main_args.enable_wandb:
            wandb.config.update(retriever_args, allow_val_change=False)
        logger.info("Indexing is starting...")
        retriever = Retriever(embedder.embedding_dim, index_type=retriever_args.index_type,
                              normalize_index=retriever_args.normalize_index)
        retriever.add_embeddings(embeddings, meta_datas)
        retriever.save_index(retriever_args.index_path)
        logger.info("Indexing is done...")

    retrieved_dataset = None
    if main_args.do_searching:
        if main_args.enable_wandb:
            wandb.config.update(retriever_args, allow_val_change=False)
        logger.info("Loading retriever's index...")
        retriever = Retriever()
        retriever.load_index(retriever_args.index_path)
        logger.info("Retrieving similar sentences...")

        if retriever_args.k == 0:
            retriever_args.k = max((retriever_args.max_retrieved // len(embeddings)), 1) * 50
        if retrieval_tuner_args.random_retrieve:
            retrieved_data = retriever.retrieve_random_metadata(max_retrieved=retriever_args.max_retrieved,
                                                exclude_datasets=retriever_args.exclude_datasets,
                                                exclude_languages=retriever_args.exclude_languages,
                                                unique_word_criteria_weight=retriever_args.unique_word_criteria_weight,
                                                cluster_criteria_weight=retriever_args.cluster_criteria_weight,
                                                balance_labels=retriever_args.balance_labels,)
        else:
            retrieved_data = retriever.retrieve_multiple_queries(
                query_embeddings=embeddings,
                k=retriever_args.k,
                max_retrieved=retriever_args.max_retrieved,
                exclude_datasets=retriever_args.exclude_datasets,
                exclude_languages=retriever_args.exclude_languages,
                unique_word_criteria_weight=retriever_args.unique_word_criteria_weight,
                cluster_criteria_weight=retriever_args.cluster_criteria_weight,
                balance_labels=retriever_args.balance_labels,
            )
        retriever.save_meta_to_file(retrieved_data, finetuner_args.output_dir)
        logger.info("Retrieved %d instances based on query.", len(retrieved_data))

        # Convert retrieved data to dataset format
        retrieved_dataset = data_provider.convert_to_dataset(retrieved_data)

    if embedder:
        # Free GPU memory by deleting the model and calling garbage collection
        del embedder.model
        del embedder
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Embedding model deleted and GPU memory cleared.")

    if not main_args.do_searching and retrieval_tuner_args.combine_train_set:
        dataset = data_provider.aggregate_splits([dataset['data'] for dataset in datasets], just_aggregate=['train'])
    else:
        dataset = data_provider.aggregate_splits([dataset['data'] for dataset in datasets])

    if main_args.do_searching and retrieval_tuner_args.combine_train_set:
        dataset = data_provider.combine_new_dataset(dataset, retrieved_dataset,
                                                 repeat=finetuner_args.repeat_target_train_set)

    retrieval_tuning_model_path = ''
    if main_args.do_retrieval_tuning:
        if main_args.enable_wandb:
            wandb.config.update(finetuner_args, allow_val_change=False)
        retrieval_tuner_args = copy_args(retrieval_tuner_args, finetuner_args)
        retrieval_tuner = FineTuner(retrieval_tuner_args)
        logger.info("Retrieval fine-tuning the model: %s", retrieval_tuner_args.finetuner_model_name_or_path)

        if retrieval_tuner_args.do_train:
            # train_dataset = retrieval_tuner.prepare_data(retrieved_dataset)
            # eval_dataset = retrieval_tuner.prepare_data(dataset['validation'])

            retrieval_tuner.train(
                retrieved_dataset, dataset['validation']
            )
            logger.info("Retrieval fine-tuning on retrieved instances completed.")
        if retrieval_tuner_args.do_test:
            # test_dataset = retrieval_tuner.prepare_data(dataset['test'])
            results = retrieval_tuner.evaluate(dataset['test'], save_results=True,
                                               key='retrieval_finetuner', metric_key_prefix='retrieval_finetuner')
            # results = {'fine_tuner_'+i: j for i, j in results.items()}
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Retrieval finetune based inference metrics: %s", results)
        retrieval_tuning_model_path = retrieval_tuner.save_model()
        # logger.info("First-stage fine-tuned model saved.")

        # Free GPU memory by deleting the model and calling garbage collection
        del retrieval_tuner.model
        del retrieval_tuner
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("First-trained model deleted and GPU memory cleared.")

    # Step 4: Fine-tune the model
    if main_args.do_fine_tuning:
        if main_args.enable_wandb:
            wandb.config.update(finetuner_args, allow_val_change=False)
        fine_tuner = FineTuner(finetuner_args)
        if main_args.do_retrieval_tuning and retrieval_tuning_model_path:
            fine_tuner.finetuner_model_name_or_path = retrieval_tuning_model_path
            logger.info("Continuing fine-tuning the model: %s", retrieval_tuning_model_path)
        else:
            logger.info("Fine-tuning the model: %s", finetuner_args.finetuner_model_name_or_path)


        if finetuner_args.do_train:
            # train_dataset = fine_tuner.prepare_data()
            # eval_dataset = fine_tuner.prepare_data()

            fine_tuner.train(
                dataset['train'], dataset['validation']
            )

            logger.info("Fine-tuning completed.")

        if finetuner_args.do_test:
            # test_dataset = fine_tuner.prepare_data(dataset['test'])
            results = fine_tuner.evaluate(dataset['test'], save_results=True)
            # results = {'fine_tuner_'+i: j for i, j in results.items()}
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference metrics: %s", results)

        if finetuner_args.do_hate_check and 'hate_check' in dataset:
            # test_dataset = fine_tuner.prepare_data(dataset['hate_check'])
            results = fine_tuner.evaluate(dataset['hate_check'], save_results=True, key='hate_check', metric_key_prefix='hate_check')
            # results = {'fine_tuner_'+i: j for i, j in results.items()}
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference metrics: %s", results)

    # Step 5: Prompt-based inference
    if main_args.do_prompting:
        if main_args.enable_wandb:
            wandb.config.update(prompter_args, allow_val_change=False)
        prompter = Prompter(prompter_args)
        logger.info("Running prompt-based inference with model: %s", prompter_args.prompter_model_name_or_path)
        for i, data in enumerate(datasets):
            dataset = data['data']
            if not prompter_args.prompt:
                prompter_args.prompt = prompter.form_prompt_template(language=data['language'])
            predictions = prompter.evaluate(
                test_data=dataset['test'],
                prompt_template=prompter_args.prompt
            )
            results = prompter.compute_metrics(predictions, dataset['test']['label'])
            prompter.save_results(predictions, dataset, results, name=data['name']+'_with_translated_prompt')
            # results = {'prompter_'+i: j for i, j in results.items()}
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Prompt-based inference metrics for %s is: %s", data['name'],results)

    # Finish Wandb
    if main_args.enable_wandb:
        wandb.finish()
    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()