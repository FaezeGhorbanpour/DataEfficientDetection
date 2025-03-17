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
from typing import Optional, List, Union
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
        default=256,
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
    do_hate_day: bool = field(
        default=False,
        metadata={"help": "Run the hate day evaluation."}
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
    do_early_stopping: bool=field(
        default=False,
        metadata={"help": "Set true to early stopping."}
    )
    use_step_trainer: bool=field(
        default=False,
        metadata={"help": ""}
    )
    load_best_model_at_end: bool=field(
        default=True,
        metadata={"help": ""}
    )
    label_smoothing_factor: float=field(
        default=0.0,
        metadata={"help": ""}
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
    perplexity_model: Optional[str] = field(
        default="openai-community/gpt2-large",
        metadata={"help": "Perplexity model name or path."},
    )
    add_uncertainty: bool = field(
        default=False,
        metadata={"help": "Add uncertainty."}
    )
    uncertainty_model: Optional[str] = field(
        default="cardiffnlp/twitter-roberta-base-hate",
        metadata={"help": "Uncertainty model name or path."},
    )
    uncertainty_tokenizer: Optional[str] = field(
        default="cardiffnlp/twitter-roberta-base-hate",
        metadata={"help": "Uncertainty model name or path."},
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
    include_languages: Optional[List[str]] = field(
        default=None, metadata={"help": "Only include these languages in retrieved data."}
    )
    include_datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "Only include these datasets in retrieval."}
    )
    index_type: str = field(
        default="HNSW",
        metadata={"help": "index_type"},
    )
    index_path: str = field(
        default="./index_path",
        metadata={"help": "Directory to save index path!"},
    )
    num_retrieved: int = field(
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
    perplexity_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for perplexity criteria to score retrieved embeddings."},
    )
    uncertainty_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for uncertainty criteria to score retrieved embeddings."},
    )
    margin_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for margin criteria to score retrieved embeddings."},
    )
    balance_labels: bool = field(
        default=False,
        metadata={"help": "Balance the retrieved embeddings."},
    )
    mmr_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold for the MMR."},
    )
    random_retrieve: bool = field(
        default=False,
        metadata={"help": "Combine retrieved data with training set."}
    )
    retrieve_per_instance: bool = field(
        default=False,
        metadata={"help": ""}
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
        default=256,
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





@dataclass
class PrompterArguments:
    prompter_model_name_or_path: str = field(
        default="mt0", metadata={"help": "Instruction-tuned model name or path."}
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
        default=None,
        metadata={"help": "Batch size for training and evaluation."}
    )
    prompter_max_length: int = field(
        default=None,
        metadata={"help": "."}
    )
    prompts_list: Union[List[str], str] = field(
        default_factory=lambda: "all",  # Default value can be "all", "none", or a list
        metadata={"help": "A list of prompts, or the special values 'none' or 'all'."}
    )
    num_rounds: int = field(
        default=3,
        metadata={"help": "Number of prompting rounds."}
    )
    do_zero_shot_prompting: bool = field(
        default=True,
        metadata={"help": "Do zero shot prompting."}
    )
    do_few_shot_prompting: bool = field(
        default=False,
        metadata={"help": "Do few shot prompting."}
    )


    # max_length: int = field(
    #     default=256,
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
    run_optuna: bool = field(
        default=False,
        metadata={"help": "Whether to run Optuna hyperparameter optimization"}
    )
    optuna_n_trials: int = field(
        default=20,
        metadata={"help": "Number of trials for Optuna optimization"}
    )
    run_best_params: bool = field(
        default=False,
        metadata={"help": "Whether to run with the best parameters after Optuna optimization"}
    )
    optuna_study_name: str = field(
        default='',
        metadata={"help": "Which study to run"}
    )
    optuna_storage_path: str=field(
        default='',
        metadata={"help": "Which storage to use"}
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



def main(
    main_args=None,
    data_args=None,
    embedder_args=None,
    retriever_args=None,
    retrieval_tuner_args=None,
    finetuner_args=None,
    prompter_args=None):

    if main_args is None:
        parser = HfArgumentParser((MainArguments, DataArguments, EmbedderArguments, RetrieverArguments,
                                   RetrievalTunerArguments, FineTunerArguments, PrompterArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            main_args, data_args, embedder_args, retriever_args, retrieval_tuner_args, finetuner_args, prompter_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]))
        else:
            main_args, data_args, embedder_args, retriever_args, retrieval_tuner_args, finetuner_args, prompter_args = parser.parse_args_into_dataclasses()

    if main_args.do_fine_tuning:
        file_path = os.path.join(finetuner_args.output_dir, "evaluation_results.json")
        if os.path.exists(file_path):
            print(f"Error: The file {file_path} exist. Aborting the run.")
            sys.exit(1)  # Abort the run

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
        wandb.config.update(data_args, allow_val_change=True)

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
        embedder = Embedder(embedder_args.embedder_model_name_or_path,
                            add_perplexity=embedder_args.add_perplexity,
                            perplexity_model=embedder_args.perplexity_model,
                            add_uncertainty=embedder_args.add_uncertainty,
                            uncertainty_model=embedder_args.uncertainty_model,
                            uncertainty_tokenizer=embedder_args.uncertainty_tokenizer)
        if main_args.enable_wandb:
            wandb.config.update(embedder_args, allow_val_change=True)
        logger.info("Embedding datasets...")

        embeddings, meta_datas = embedder.embed_datasets(datasets, splits=embedder_args.splits)
        logger.info("Datasets embedded with model: %s", embedder_args.embedder_model_name_or_path)




    # Step 3: Retrieve similar sentences
    if main_args.do_indexing:
        if main_args.enable_wandb:
            wandb.config.update(retriever_args, allow_val_change=True)
        logger.info("Indexing is starting...")
        retriever = Retriever(embedder.embedding_dim, index_type=retriever_args.index_type,
                              normalize_index=retriever_args.normalize_index)
        retriever.add_embeddings(embeddings, meta_datas,
                mmr_threshold=retriever_args.mmr_threshold)
        retriever.save_index(retriever_args.index_path)
        logger.info("Indexing is done...")

    retrieved = None
    if main_args.do_searching:
        if main_args.enable_wandb:
            wandb.config.update(retriever_args, allow_val_change=True)
        logger.info("Loading retriever's index...")
        retriever = Retriever(embedder.embedding_dim, index_type=retriever_args.index_type,
                              normalize_index=retriever_args.normalize_index)
        retriever.load_index(retriever_args.index_path)
        logger.info("Retrieving similar sentences...")

        if retriever_args.k == 0:
            import math
            retriever_args.k = max(math.ceil(3 * retriever_args.num_retrieved / len(embeddings)),
                                   math.ceil(2000 / len(embeddings)), 2)
        retrieved = None
        if retriever_args.random_retrieve:
            retrieved = retriever.retrieve_random_metadata(num_retrieved=retriever_args.num_retrieved,
                                                exclude_datasets=retriever_args.exclude_datasets,
                                                exclude_languages=retriever_args.exclude_languages,
                                                unique_word_criteria_weight=retriever_args.unique_word_criteria_weight,
                                                cluster_criteria_weight=retriever_args.cluster_criteria_weight,
                                                balance_labels=retriever_args.balance_labels,)
        elif retriever_args.retrieve_per_instance:
            retrieved = dict()
            for i in range(embeddings.shape[0]):
                # id = meta_datas[i]['dataset_name'] +'-'+ meta_datas[i]['id'] +'-'+ meta_datas[i]['split']
                id = meta_datas[i]['id']
                retrieved[id] = retriever.retrieve_one_query(
                    query_embedding=embeddings[i],
                    k=retriever_args.k,
                    unique_word_criteria_weight=retriever_args.unique_word_criteria_weight,
                    cluster_criteria_weight=retriever_args.cluster_criteria_weight,
                    perplexity_weight=retriever_args.perplexity_weight,
                    uncertainty_weight=retriever_args.uncertainty_weight,
                    margin_weight=retriever_args.margin_weight,
                    balance_labels=retriever_args.balance_labels,
                    mmr_threshold=retriever_args.mmr_threshold
                )

            retriever.save_meta_to_file(retrieved, finetuner_args.output_dir+'/retrieved_data/'+meta_datas[0]['dataset_name'])
            logger.info("Retrieved %d instances based on query.", len(retrieved))
        else:
            retrieved = retriever.retrieve_multiple_queries(
                query_embeddings=embeddings,
                k=retriever_args.k,
                num_retrieved=retriever_args.num_retrieved,
                exclude_datasets=retriever_args.exclude_datasets,
                exclude_languages=retriever_args.exclude_languages,
                unique_word_criteria_weight=retriever_args.unique_word_criteria_weight,
                cluster_criteria_weight=retriever_args.cluster_criteria_weight,
                perplexity_weight=retriever_args.perplexity_weight,
                uncertainty_weight=retriever_args.uncertainty_weight,
                margin_weight=retriever_args.margin_weight,
                balance_labels=retriever_args.balance_labels,
                mmr_threshold=retriever_args.mmr_threshold
            )
            retriever.save_meta_to_file(retrieved, finetuner_args.output_dir)
            logger.info("Retrieved %d instances based on query.", len(retrieved))

    if embedder:
        # Free GPU memory by deleting the model and calling garbage collection
        del embedder.model
        del embedder
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Embedding model deleted and GPU memory cleared.")

    if main_args.do_searching and (main_args.do_fine_tuning or main_args.do_retrieval_tuning):
        # Convert retrieved data to dataset format
        retrieved_dataset = data_provider.convert_to_dataset(retrieved)
        if retrieval_tuner_args.combine_train_set:
            dataset = data_provider.aggregate_splits([dataset['data'] for dataset in datasets])
            dataset = data_provider.combine_new_dataset(dataset, retrieved_dataset,
                                                        repeat=finetuner_args.repeat_target_train_set)
    elif not main_args.do_searching and (main_args.do_fine_tuning and retrieval_tuner_args.combine_train_set):
        #TODO Why? For baseline and rotger's combine train sets experiments
        dataset = data_provider.aggregate_splits([dataset['data'] for dataset in datasets], just_aggregate=['train'])
    else:
        dataset = data_provider.aggregate_splits([dataset['data'] for dataset in datasets])
    shots = None
    if main_args.do_searching and main_args.do_prompting:
        shots = data_provider.extract_text_and_label(retrieved)


    retrieval_tuning_model_path = ''
    if main_args.do_retrieval_tuning:
        if main_args.enable_wandb:
            wandb.config.update(finetuner_args, allow_val_change=True)
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
            wandb.config.update(finetuner_args, allow_val_change=True)
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

        if finetuner_args.do_eval:
            results = fine_tuner.evaluate(dataset['validation'], save_results=True, key='validation', metric_key_prefix='validation')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on eval dataset metrics: %s", results)

            if main_args.run_optuna:
                return results['validation_f1-macro']

        if finetuner_args.do_test:
            results = fine_tuner.evaluate(dataset['test'], save_results=True)
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on test dataset metrics: %s", results)

        if finetuner_args.do_hate_check and 'hate_check' in dataset:
            results = fine_tuner.evaluate(dataset['hate_check'], save_results=True, key='hate_check',
                                          metric_key_prefix='hate_check')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on hate-check metrics: %s", results)

        if finetuner_args.do_hate_day and 'hate_day' in dataset:
            results = fine_tuner.evaluate(dataset['hate_day'], save_results=True, key='hate_day',
                                          metric_key_prefix='hate_day')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on hate-day metrics: %s", results)


    # Step 5: Prompt-based inference
    if main_args.do_prompting:
        if main_args.enable_wandb:
            wandb.config.update(prompter_args, allow_val_change=True)
        prompter = Prompter(prompter_args)
        logger.info("Running prompt-based inference with model: %s", prompter_args.prompter_model_name_or_path)
        for i, data in enumerate(datasets):
            prompter.do_zero_shot_prompting(data)
            logger.info("Prompt-based zero-shot inference metrics for %s finished. ", data['name'])
            if shots:
                prompter.do_few_shot_prompting(data, shots)
                logger.info("Prompt-based few-shot inference metrics for %s finished. ", data['name'])

    if main_args.do_prompting:
        main_args.do_prompting = False
        data_args.datasets = ["dyn21_en", "fou18_en", "ken20_en", "xplain_en", "implicit_en", "xdomain_en"]
        data_args.languages = ["en", "en", "en", "en", "en", "en"]
        main_args.do_embedding = True
        embedder_args.embedder_model_name_or_path = "m3"
        main_args.do_indexing = True
        embedder_args.add_perplexity = True
        embedder_args.add_uncertainty = True
        embedder_args.splits = ["train", "validation", "test"]
        retriever_args.mmr_threshold = 0.80
        retriever_args.index_path = "/mounts/data/proj/faeze/data_efficient_hate/models/retriever/en_m3_HNSW-mmr/"
        finetuner_args.output_path = "/mounts/data/proj/faeze/data_efficient_hate/models/retriever/en_m3_HNSW-mmr/"
        main_args.wandb_run_name = "embedding_english"
        main(
            main_args,
            data_args,
            embedder_args,
            retriever_args,
            retrieval_tuner_args,
            finetuner_args,
            prompter_args)

    # Finish Wandb
    if main_args.enable_wandb:
        wandb.finish()
    logger.info("Pipeline execution completed.")


import logging
import os
import sys
from transformers import HfArgumentParser


def objective(trial, parsed_args):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting trial {trial.number}")

    # Unpack the parsed arguments
    main_args, data_args, embedder_args, retriever_args, retrieval_tuner_args, finetuner_args, prompter_args = parsed_args

    # Modify arguments with Optuna suggestions
    # The 4 requested parameters
    finetuner_args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    finetuner_args.num_train_epochs = trial.suggest_int("num_epochs", 2, 30)
    finetuner_args.weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    finetuner_args.max_sequence_length = trial.suggest_categorical("max_sequence_length", [128, 256, 512])

    logger.info(f"Trial {trial.number} parameters: "
                f"lr={finetuner_args.learning_rate}, "
                f"epochs={finetuner_args.num_train_epochs}, "
                f"weight_decay={finetuner_args.weight_decay}, "
                f"max_seq_len={finetuner_args.max_sequence_length}")

    # finetuner_args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    # finetuner_args.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # finetuner_args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    # finetuner_args.use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
    # finetuner_args.lr_scheduler = trial.suggest_categorical("lr_scheduler",
    #                                                         ["linear", "cosine", "cosine_with_restarts"])

    # Call main with the modified arguments
    try:
        logger.info(f"Running main function with trial {trial.number}")
        macro_f1 = main(main_args=main_args, data_args=data_args, embedder_args=embedder_args,
                        retriever_args=retriever_args,
                        retrieval_tuner_args=retrieval_tuner_args, finetuner_args=finetuner_args,
                        prompter_args=prompter_args)
        logger.info(f"Trial {trial.number} completed with macro_f1={macro_f1}")
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}", exc_info=True)
        raise

    return macro_f1


if __name__ == "__main__":
    # Set up logging
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting optimization script")

    # Parse arguments first
    logger.info("Parsing arguments")
    parser = HfArgumentParser((MainArguments, DataArguments, EmbedderArguments, RetrieverArguments,
                               RetrievalTunerArguments, FineTunerArguments, PrompterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        parsed_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        logger.info(f"Parsed arguments from JSON file: {sys.argv[1]}")
    else:
        parsed_args = parser.parse_args_into_dataclasses()
        logger.info("Parsed arguments from command line")

    # Check if we should run hyperparameter tuning
    main_args = parsed_args[0]
    if hasattr(main_args, "run_optuna") and main_args.run_optuna:
        # Run Optuna optimization
        logger.info("Starting Optuna hyperparameter optimization")
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner

        # Set up Optuna storage directory
        os.makedirs(main_args.optuna_storage_path, exist_ok=True)
        logger.info(f"Created Optuna storage directory: {main_args.optuna_storage_path}")

        n_trials = main_args.optuna_n_trials if hasattr(main_args, "optuna_n_trials") else 20
        logger.info(f"Will run {n_trials} trials")

        db_path = f'sqlite:///{main_args.optuna_storage_path + "/optuna.db"}'
        logger.info(f"Using database path: {db_path}")

        # Create and configure study
        # try:
        study = optuna.create_study(
            study_name=main_args.optuna_study_name,
            sampler=TPESampler(),
            storage=db_path,
            load_if_exists=True,
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        logger.info(f"Created/loaded study '{main_args.optuna_study_name}'")

        # Add Optuna logging callback
        optuna_logger = optuna.logging.get_logger("optuna")
        optuna_logger.addHandler(logging.FileHandler(os.path.join(main_args.optuna_storage_path, "optuna_internal.log")))

        # Run optimization
        logger.info("Starting optimization process")
        study.optimize(lambda trial: objective(trial, parsed_args), n_trials=n_trials)

        logger.info("Optimization completed")
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

        # Optionally run with the best parameters
        if hasattr(main_args, "run_best_params") and main_args.run_best_params:
            logger.info("Running with best parameters")
            # Unpack parsed arguments
            main_args, data_args, embedder_args, retriever_args, retrieval_tuner_args, finetuner_args, prompter_args = parsed_args

            # Update with best parameters
            for key, value in trial.params.items():
                logger.info(f"Setting best parameter {key}={value}")
                if key.startswith("retriever_"):
                    setattr(retriever_args, key.replace("retriever_", ""), value)
                elif key.startswith("embedder_"):
                    setattr(embedder_args, key.replace("embedder_", ""), value)
                elif key.startswith("finetuner_"):
                    setattr(embedder_args, key.replace("finetuner_", ""), value)
                # Add more conditions for other argument types as needed

            # Run with best parameters
            logger.info("Running main with best parameters")
            main(
                main_args=main_args,
                data_args=data_args,
                embedder_args=embedder_args,
                retriever_args=retriever_args,
                retrieval_tuner_args=retrieval_tuner_args,
                finetuner_args=finetuner_args,
                prompter_args=prompter_args
            )
        # except Exception as e:
        #     logger.error(f"Error during Optuna study: {str(e)}", exc_info=True)
        #     raise
    else:
        # Run normally
        logger.info("Running in normal mode (no Optuna)")
        main()

    logger.info("Script execution completed")