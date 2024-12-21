import argparse
import logging
import wandb
from finetuner import FineTuner, FineTunerConfig
from data_provider import DataProvider
from prompter import Prompter

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Fine-tune and test models")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of dataset names.")
    parser.add_argument("--languages", nargs="+", required=True, help="List of languages.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset.")
    parser.add_argument("--fine_tune_type", type=str, default="default", help="Fine-tuning method.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--wandb_project", type=str, default="multilingual_nlp_pipeline", help="Wandb project.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to fine-tune the model.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for the inputs.")
    args = parser.parse_args()

    # Initialize Wandb
    wandb.init(project=args.wandb_project, config=vars(args))
    logger.info("Wandb initialized with config: %s", vars(args))

    # Initialize data provider and fine-tuner
    data_provider = DataProvider()
    fine_tuner_config = FineTunerConfig(
        model_name=args.model_name,
        fine_tune_method=args.fine_tune_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        # max_seq_length=args.max_seq_length
    )
    fine_tuner = FineTuner(fine_tuner_config)

    # Load datasets
    datasets = data_provider.load_datasets(args.datasets, args.languages, args.max_samples)
    logger.info("Datasets loaded: %s", args.datasets)

    train_data = datasets[0]['data']
    eval_data = datasets[1]['data']
    test_data = datasets[2]['data']

    # Prepare datasets for fine-tuning
    train_dataset = fine_tuner.prepare_data(train_data)
    eval_dataset = fine_tuner.prepare_data(eval_data)
    test_dataset = fine_tuner.prepare_data(test_data)

    # Fine-tune the model
    logger.info("Fine-tuning started.")
    fine_tuner.train(train_dataset, eval_dataset)
    logger.info("Fine-tuning completed.")

    # Evaluate the model
    prediction = fine_tuner.predict(test_dataset)
    results = fine_tuner.compute_metrics(prediction)
    logger.info("Test results: %s", results)
    wandb.log({"test_results": results})

    # Test the model with a prompter
    prompter = Prompter(args.model_name)
    logger.info("Generating predictions using prompter.")
    predictions = prompter.test(test_data, 'Classify the text to hate and non-hate. answer with "1" if it is hate or "0" if it non-hate. text: {input}')
    predictions = [int(i) for i in predictions]
    metrics = prompter.compute_metrics(predictions, test_data['label'])
    logger.info("Prompt-based test results: %s", metrics)
    wandb.log({"prompt_based_test_results": metrics})

    # Finish Wandb
    wandb.finish()
    logger.info("Fine-tuning and testing complete.")

if __name__ == "__main__":
    main()
