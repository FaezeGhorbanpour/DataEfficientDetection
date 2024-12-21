# Embedding and indexing datasets
import argparse
import logging
import wandb
from data_provider import DataProvider
from embedder import Embedder
from retriever import Retriever
from datasets import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Embedding and indexing datasets")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of dataset names.")
    parser.add_argument("--languages", nargs="+", required=True, help="List of languages.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset.")
    parser.add_argument("--embed_model", type=str, required=True, help="Embedding model name.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to save index.")
    parser.add_argument("--wandb_project", type=str, default="multilingual_nlp_pipeline", help="Wandb project.")
    args = parser.parse_args()

    # Initialize Wandb
    wandb.init(project=args.wandb_project, config=vars(args))
    logger.info("Wandb initialized")

    data_provider = DataProvider()
    embedder = Embedder(model_name=args.embed_model)
    retriever = Retriever()

    datasets = data_provider.load_datasets(args.datasets, args.languages, args.max_samples)

    embeddings, metadatas = embedder.embed_datasets(datasets)
    retriever.add_embeddings(embeddings, metadatas)

    retriever.save_index(args.index_path)
    wandb.finish()
    logger.info(f"Index saved at {args.index_path}")

if __name__ == "__main__":
    main()
