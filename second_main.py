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



# Define arguments for each module using dataclasses

@dataclass
class FirstDataArguments:
    first_languages: List[str] = field(
        default_factory=list, metadata={"help": "Languages for the datasets."}
    )
    first_datasets: List[str] = field(
        default_factory=list, metadata={"help": "List of dataset names to load."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    max_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum samples per dataset."}
    )

@dataclass
class FirstFineTunerArguments(TrainingArguments):
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
    output_dir: str = field(
        default='',
        metadata={"help": ''}
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


@dataclass
class SecondDataArguments:
    second_languages: List[str] = field(
        default_factory=list, metadata={"help": "Languages for the datasets."}
    )
    second_datasets: List[str] = field(
        default_factory=list, metadata={"help": "List of dataset names to load."}
    )
    second_dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the dataset downloaded from huggingface.co"},
    )
    second_max_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum samples per dataset."}
    )


@dataclass
class SecondFineTunerArguments:
    second_fine_tune_method: str = field(
        default="default",
        metadata={"help": "Fine-tuning method (e.g., lora, prefix)."}
    )
    second_num_train_epochs: int = field(
        default=None,
        metadata={"help": "Batch size for training and evaluation."}
    )
    second_train_batch_size: int = field(
        default=None,
        metadata={"help": "Batch size for training and evaluation."}
    )
    second_eval_batch_size: int = field(
        default=None,
        metadata={"help": "Batch size for training and evaluation."}
    )
    second_num_labels: int = field(
        default=None,
        metadata={"help": "Number of labels for classification tasks."}
    )
    second_peft_config: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for PEFT methods like LoRA or prefix tuning."}
    )
    second_max_seq_length: int = field(
        default=None,
        metadata={"help": "Limit the total number of checkpoints to save."}
    )
    do_second_train: bool = field(
        default=False,
        metadata={"help": "Set true to train the FineTuner."}
    )
    do_second_eval: bool = field(
        default=False,
        metadata={"help": "Set true to train the FineTuner."}
    )
    do_second_test: bool = field(
        default=False,
        metadata={"help": "Set true to test the FineTuner."}
    )
    do_second_hate_check: bool = field(
        default=False,
        metadata={"help": "Set true to test the FineTuner."}
    )
    do_second_hate_day: bool = field(
        default=False,
        metadata={"help": "Set true to test the FineTuner."}
    )
    combine_train_set: bool = field(
        default=False,
        metadata={"help": "Combine retrieved data with training set."}
    )
    second_output_dir: str = field(
        default='',
        metadata={'help': ''}
    )





@dataclass
class MainArguments:
    wandb_project: str = field(
        default="transferability",
        metadata={"help": "Wandb project name."}
    )
    wandb_run_name: Optional[str] = field(
        default='temp',
        metadata={"help": "Wandb run name."}
    )
    do_first_fine_tuning: bool = field(
        default=False,
        metadata={"help": "Run the fine-tuning step."}
    )
    do_second_fine_tuning: bool = field(
        default=False,
        metadata={"help": "Run the fine-tuning step."}
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


def copy_finetuner_args(second_finetuner_args, first_finetuner_args):
    """
    Create a copy of first_finetuner_args and update it with values from second_finetuner_args.
    """
    finetuner_args_copy = copy.deepcopy(first_finetuner_args)

    # Direct assignments
    finetuner_args_copy.fine_tune_method = second_finetuner_args.second_fine_tune_method
    finetuner_args_copy.peft_config = second_finetuner_args.second_peft_config
    finetuner_args_copy.do_train = second_finetuner_args.do_second_train
    finetuner_args_copy.do_eval = second_finetuner_args.do_second_eval
    finetuner_args_copy.do_test = second_finetuner_args.do_second_test
    finetuner_args_copy.do_hate_check = second_finetuner_args.do_second_hate_check
    finetuner_args_copy.do_hate_day = second_finetuner_args.do_second_hate_day
    finetuner_args_copy.report_to = []

    # Conditional assignments
    if second_finetuner_args.second_num_train_epochs:
        finetuner_args_copy.num_train_epochs = second_finetuner_args.second_num_train_epochs

    if second_finetuner_args.second_train_batch_size:
        finetuner_args_copy.per_device_train_batch_size = second_finetuner_args.second_train_batch_size

    if second_finetuner_args.second_eval_batch_size:
        finetuner_args_copy.per_device_eval_batch_size = second_finetuner_args.second_eval_batch_size

    if second_finetuner_args.second_num_labels:
        finetuner_args_copy.num_labels = second_finetuner_args.second_num_labels

    if second_finetuner_args.second_max_seq_length:
        finetuner_args_copy.max_seq_length = second_finetuner_args.second_max_seq_length

    if second_finetuner_args.second_output_dir:
        finetuner_args_copy.output_dir = second_finetuner_args.second_output_dir

    return finetuner_args_copy



def main(
    main_args=None,
    first_data_args=None,
    second_data_args=None,
    first_finetuner_args=None,
    second_finetuner_args=None
):

    if main_args is None:
        parser = HfArgumentParser((MainArguments, FirstDataArguments, SecondDataArguments,
                                   FirstFineTunerArguments, SecondFineTunerArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            main_args, first_data_args, second_data_args, first_finetuner_args, second_finetuner_args = (
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])))
        else:
            main_args, first_data_args, second_data_args, first_finetuner_args, second_finetuner_args = (
                parser.parse_args_into_dataclasses())

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if is_main_process(first_finetuner_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {first_finetuner_args.local_rank}, device: {first_finetuner_args.device}, n_gpu: {first_finetuner_args.n_gpu}"
        + f"distributed training: {bool(first_finetuner_args.local_rank != -1)}, 16-bits training: {first_finetuner_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(first_finetuner_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {first_finetuner_args}")

    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # Initialize Wandb
    if main_args.enable_wandb:
        wandb.init(
            project=main_args.wandb_project,
            name=f"{main_args.wandb_run_name}",
            config=main_args,
        )
        # wandb.config["log_frequency"] = 1000
        # wandb.config["log_model"] = False
        logger.info("Wandb initialized.")


    if first_finetuner_args.report_to is None or first_finetuner_args.report_to == 'None' or 'None' in first_finetuner_args.report_to:
        first_finetuner_args.report_to = []
    elif first_finetuner_args.report_to == 'wandb':
        first_finetuner_args.report_to = ['wandb']

    # Set seed before initializing model.
    set_seed(first_finetuner_args.seed)

    # Initialize modules
    data_provider = DataProvider()


    #########################################################################
    first_fine_tuning_model_path = ''
    if main_args.do_first_fine_tuning:
        file_path = os.path.join(first_finetuner_args.output_dir, "evaluation_results.json")
        if os.path.exists(file_path):
            print(f"Error: The file {file_path} exist. Aborting the first run.")
            main_args.do_first_fine_tuning = False
            first_fine_tuning_model_path = os.path.join(first_finetuner_args.output_dir, "the_best_checkpoint")

    if main_args.do_first_fine_tuning:
        # Step 1: Load first datasets
        logger.info(f"Loading datasets: {first_data_args.first_datasets} ...")

        if len(first_data_args.first_datasets) > 1:
            first_datasets = data_provider.load_datasets(
                dataset_names=first_data_args.first_datasets,
                languages=first_data_args.first_languages,
                cache_dir=first_data_args.dataset_cache_dir,
                max_samples=first_data_args.max_samples
            )
            first_dataset = data_provider.aggregate_splits(first_datasets)
        else:
            first_dataset = data_provider.load_dataset(
                dataset_name=first_data_args.first_datasets[0],
                cache_dir=first_data_args.dataset_cache_dir,
                max_samples=first_data_args.max_samples
            )
        logger.info("First datasets loaded.")

        if main_args.enable_wandb:
            wandb.config.update(first_finetuner_args, allow_val_change=True)
            wandb.config.update(first_data_args, allow_val_change=True)

        first_tuner = FineTuner(first_finetuner_args)
        logger.info("Retrieval fine-tuning the model: %s", first_finetuner_args.finetuner_model_name_or_path)

        if first_finetuner_args.do_train:

            first_tuner.train(
                first_dataset['train'], first_dataset['validation']
            )
            logger.info("First fine-tuning completed.")

        if first_finetuner_args.do_eval:
            results = first_tuner.evaluate(first_dataset['validation'], save_results=True, key='validation',
                                          metric_key_prefix='validation')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on eval dataset metrics: %s", results)

        if first_finetuner_args.do_test:
            results = first_tuner.evaluate(first_dataset['test'], save_results=True)
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on test dataset metrics: %s", results)

        if first_finetuner_args.do_hate_check and 'hate_check' in first_dataset:
            results = first_tuner.evaluate(first_dataset['hate_check'], save_results=True, key='hate_check',
                                          metric_key_prefix='hate_check')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on hate-check metrics: %s", results)

        if first_finetuner_args.do_hate_day and 'hate_day' in first_dataset:
            results = first_tuner.evaluate(first_dataset['hate_day'], save_results=True, key='hate_day',
                                          metric_key_prefix='hate_day')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on hate-day metrics: %s", results)

        first_fine_tuning_model_path = first_tuner.save_model()

        # Free GPU memory by deleting the model and calling garbage collection
        del first_tuner.model
        del first_tuner
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("First-trained model deleted and GPU memory cleared.")

    ############################################################################

    if main_args.do_second_fine_tuning:
        file_path = os.path.join(second_finetuner_args.second_output_dir, "evaluation_results.json")
        if os.path.exists(file_path):
            print(f"Error: The file {file_path} exist. Aborting the run.")
            main_args.do_second_fine_tuning = False

    # Step 2: Fine-tune the model
    if main_args.do_second_fine_tuning:
        logger.info(f"Loading datasets: {second_data_args.second_datasets} ...")
        if len(second_data_args.second_datasets) > 1:
            second_datasets = data_provider.load_datasets(
                dataset_names=second_data_args.second_datasets,
                languages=second_data_args.second_languages,
                cache_dir=second_data_args.second_dataset_cache_dir,
                max_samples=second_data_args.second_max_samples
            )
            second_dataset = data_provider.aggregate_splits(second_datasets)
        else:
            second_dataset = data_provider.load_dataset(
                dataset_name=second_data_args.second_datasets[0],
                cache_dir=second_data_args.second_dataset_cache_dir,
                max_samples=second_data_args.second_max_samples
            )
        logger.info("Second datasets loaded.")


        if main_args.enable_wandb:
            wandb.config.update(second_finetuner_args, allow_val_change=True)
            wandb.config.update(second_data_args, allow_val_change=True)

        second_finetuner_args = copy_finetuner_args(second_finetuner_args, first_finetuner_args)
        if first_fine_tuning_model_path:
            second_finetuner_args.finetuner_model_name_or_path = first_fine_tuning_model_path
            second_finetuner_args.finetuner_tokenizer_name_or_path = first_finetuner_args.finetuner_model_name_or_path
            logger.info("Continuing fine-tuning the model: %s", first_fine_tuning_model_path)
        else:
            logger.info("Fine-tuning the model: %s", second_finetuner_args.finetuner_model_name_or_path)

        second_tuner = FineTuner(second_finetuner_args)

        if second_finetuner_args.do_train:
            second_tuner.train(
                second_dataset['train'], second_dataset['validation']
            )

            logger.info("Second Fine-tuning completed.")

        if second_finetuner_args.do_eval:
            results = second_tuner.evaluate(second_dataset['validation'], save_results=True, key='validation',
                                          metric_key_prefix='validation')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on eval dataset metrics: %s", results)

            if main_args.run_optuna:
                return results['validation_f1-macro']

        if second_finetuner_args.do_test:
            results = second_tuner.evaluate(second_dataset['test'], save_results=True)
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on test dataset metrics: %s", results)

        if second_finetuner_args.do_hate_check and 'hate_check' in second_dataset:
            results = second_tuner.evaluate(second_dataset['hate_check'], save_results=True, key='hate_check',
                                          metric_key_prefix='hate_check')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on hate-check metrics: %s", results)

        if second_finetuner_args.do_hate_day and 'hate_day' in second_dataset:
            results = second_tuner.evaluate(second_dataset['hate_day'], save_results=True, key='hate_day',
                                          metric_key_prefix='hate_day')
            if main_args.enable_wandb:
                wandb.log(results)
            logger.info("Finetune-based inference on hate-day metrics: %s", results)



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
    main_args, first_data_args, second_data_args, first_finetuner_args, second_finetuner_args = parsed_args

    # Modify arguments with Optuna suggestions
    # The 4 requested parameters
    first_finetuner_args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    first_finetuner_args.num_train_epochs = trial.suggest_int("num_epochs", 2, 30)
    first_finetuner_args.weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    first_finetuner_args.max_sequence_length = trial.suggest_categorical("max_sequence_length", [128, 256, 512])
    #
    # logger.info(f"Trial {trial.number} parameters: "
    #             f"lr={first_finetuner_args.learning_rate}, "
    #             f"epochs={first_finetuner_args.num_train_epochs}, "
    #             f"weight_decay={first_finetuner_args.weight_decay}, "
    #             f"max_seq_len={first_finetuner_args.max_sequence_length}")

    # first_finetuner_args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    # first_finetuner_args.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # first_finetuner_args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    # first_finetuner_args.use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
    # first_finetuner_args.lr_scheduler = trial.suggest_categorical("lr_scheduler",
    #                                                         ["linear", "cosine", "cosine_with_restarts"])

    # Call main with the modified arguments
    try:
        logger.info(f"Running main function with trial {trial.number}")
        macro_f1 = main(main_args=main_args, first_data_args=first_data_args, second_data_args=second_data_args,
                        first_finetuner_args=first_finetuner_args, second_finetuner_args=second_finetuner_args)
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
    parser = HfArgumentParser((MainArguments, FirstDataArguments, SecondDataArguments, FirstFineTunerArguments,
                               SecondFineTunerArguments))
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
            main_args, first_data_args, second_data_args, first_finetuner_args, second_finetuner_args = parsed_args

            # Update with best parameters
            for key, value in trial.params.items():
                logger.info(f"Setting best parameter {key}={value}")
                if key.startswith("first_"):
                    setattr(first_finetuner_args, key.replace("first_", ""), value)
                elif key.startswith("second_"):
                    setattr(second_finetuner_args, key.replace("second_", ""), value)
                # Add more conditions for other argument types as needed

            # Run with best parameters
            logger.info("Running main with best parameters")
            main(
                main_args=main_args,
                first_data_args=first_data_args,
                second_data_args=second_data_args,
                first_finetuner_args=first_finetuner_args,
                second_finetuner_args=second_finetuner_args
            )
        # except Exception as e:
        #     logger.error(f"Error during Optuna study: {str(e)}", exc_info=True)
        #     raise
    else:
        # Run normally
        logger.info("Running in normal mode (no Optuna)")
        main()

    logger.info("Script execution completed")
