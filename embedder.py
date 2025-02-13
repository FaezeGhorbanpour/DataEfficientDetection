import logging
import os.path

import torch
import umap
import pandas as pd
# import datashader as ds
# import datashader.transfer_functions as tf
# import colorcet as cc
# from datashader.mpl_ext import dsshow
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from preplixity_calculator import PerplexityCalculator

logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name, device="cuda", batch_size=256, add_perplexity=False):
        """
        Initialize the Embedder with a multilingual model.
        Args:
            model_name (str): Name of the Hugging Face model to load.
            device (str): The device to run the model on ('cuda' or 'cpu').
            batch_size (int): Batch size for embeddings.
        """
        logger.info(f"Initializing Embedder with model: {model_name}")
        self.name = model_name.lower()
        self.device = device
        self.batch_size = batch_size

        # Default model initialization
        self.model = None
        self.tokenizer = None
        self.type = "sentence_transformers"

        self.add_perplexity = add_perplexity
        if add_perplexity:
            self.perplexity_calculator = PerplexityCalculator(model_name="facebook/xglm-564M", batch_size=2, device=device)

        model_mapping = {
            'labse': 'sentence-transformers/LaBSE',
            'minilm': "paraphrase-multilingual-MiniLM-L12-v2",
            'm3': "BAAI/bge-m3",
            'm3-unsupervised': "BAAI/bge-m3-unsupervised",
            'm3-retromae': "BAAI/bge-m3-retromae",
            'e5-large': 'intfloat/multilingual-e5-large',
            'e5-base': 'intfloat/multilingual-e5-base',
            'arctic-large': 'Snowflake/snowflake-arctic-embed-l-v2.0',
            'arctic-base': 'Snowflake/snowflake-arctic-embed-m',
            'mpnet': 'paraphrase-multilingual-mpnet-base-v2',
            'distiluse': 'distiluse-base-multilingual-cased-v2',
            'nomic': 'nomic-ai/nomic-embed-text-v2-moe',
        }

        if self.name in model_mapping:
            self.model = SentenceTransformer(model_mapping[self.name], trust_remote_code=True)
        elif self.name == 'xlmr-large':
            self._initialize_transformer_model("FacebookAI/xlm-roberta-large")
        elif self.name == 'xlmr-base':
            self._initialize_transformer_model("FacebookAI/xlm-roberta-base")
        elif self.name == 'sonar':
            self._initialize_transformer_model("facebook/sonar")
        else:
            # Load the model as a custom SentenceTransformer
            self.model = SentenceTransformer(model_name)


        if self.type == 'sentence_transformers':
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = self.model.config.hidden_size
            self.model = self.model.to(device)
        logger.info(f"Model initialized: {model_name} with embedding dimension {self.embedding_dim}")


    def _initialize_transformer_model(self, model_name):
        """
        Helper method to initialize a transformer model and tokenizer.
        Args:
            model_name (str): Name of the transformer model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.type = "transformers"

    def embed_sentences(self, sentences):
        """
        Efficiently embed a list of sentences using batch processing.
        Args:
            sentences (list[str]): Sentences to embed.
        Returns:
            np.ndarray: Embedding vectors.
        """
        logger.info(f"Embedding {len(sentences)} sentences")

        # Process in batches
        embeddings = []
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Embedding sentences"):
            batch = sentences[i:i + self.batch_size]

            if self.type == 'transformers':
                # Tokenize and pass through SONAR in batches
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
                    self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
            else:
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                batch_embeddings = batch_embeddings.cpu().detach().numpy()

            embeddings.append(batch_embeddings)

        # Stack all embeddings
        embeddings = np.vstack(embeddings)
        logger.info("Sentences embedded successfully")
        return embeddings

    def embed_datasets(self, datasets, splits=['train', 'validation'], stack=True):
        """
        Embed all instances in multiple datasets.
        Args:
            datasets (list[dict]): Datasets with text to embed.
            splits (list[str]): What split of Datasets to be embedded.
        Returns:
            np.ndarray, list[dict]: Embeddings and corresponding metadata.
        """
        logger.info(f"Embedding datasets for splits: {splits}")
        embeddings = []
        metadata = []
        for dataset in datasets:
            logger.info(f"Processing dataset: {dataset['name']} in language {dataset['language']}")
            for split in splits:
                data = dataset["data"][split]
                texts = data["text"]
                labels = data["label"]
                ids = data["id"]
                logger.info(f"Embedding {len(texts)} texts from split: {split}")
                embs = self.embed_sentences(texts)
                embeddings.append(embs)
                metadata += [{"text": texts[i],
                              "label": labels[i],
                              "id": ids[i],
                              "split": split,
                              "dataset_name": dataset["name"],
                              "language": dataset["language"]} for i in range(len(embs))]

                if self.add_perplexity:
                    perplexities = self.perplexity_calculator.calculate_perplexity_batch(texts)
                    metadata = [
                        {**meta,
                         "perplexity": perplexities[i],
                         }
                        for i, meta in enumerate(metadata)
                    ]
        if stack:
            logger.info("Stacking all embeddings")
            return np.vstack(embeddings), metadata
        else:
            logger.info("Returning embeddings without stacking")
            return embeddings, metadata

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        """
        logger.info("Calculating similarity between two embeddings")
        norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)

        normalized1 = embedding1 / norm1
        normalized2 = embedding2 / norm2

        similarity_matrix = np.dot(normalized1, normalized2.T)
        diagonal = similarity_matrix.diagonal()
        lower_tri = np.tril(similarity_matrix)
        logger.info("Similarity calculation completed")
        return similarity_matrix, np.mean(diagonal) - np.mean(lower_tri)

    @staticmethod
    def cluster_embeddings(embeddings, metadata, n_clusters):
        """
        Cluster embeddings and return clusters with metadata.
        Args:
            embeddings (np.ndarray): Embeddings to cluster.
            metadata (list[dict]): Metadata for each embedding.
            n_clusters (int): Number of clusters.
        Returns:
            pd.DataFrame: Clustered data with metadata.
        """
        logger.info(f"Clustering embeddings into {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Create DataFrame with clusters and metadata
        data_with_clusters = pd.DataFrame(metadata)
        data_with_clusters["cluster"] = cluster_labels
        logger.info("Clustering completed")
        return data_with_clusters

    @staticmethod
    def visualize_embeddings(embeddings, metadata, reduction_method="umap", plot_title="Embedding Visualization", output_dir=None):
        """
        Visualize embeddings in 2D with Plotly and metadata as hypertext.
        Args:
            embeddings (np.ndarray): Embeddings to visualize.
            metadata (list[dict]): Metadata for each embedding.
            reduction_method (str): Dimension reduction method ("umap" or "tsne").
            plot_title (str): Title for the plot.
        Returns:
            None: Displays the plot.
        """
        logger.info(f"Visualizing embeddings using {reduction_method}")
        if reduction_method == "umap":
            reducer = umap.UMAP(random_state=42, metric='cosine')
        elif reduction_method == "tsne":
            reducer = TSNE(random_state=42)
        else:
            logger.error("Invalid reduction method. Use 'umap' or 'tsne'.")
            raise ValueError("Invalid reduction method. Use 'umap' or 'tsne'.")

        reduced_embeddings = reducer.fit_transform(embeddings)
        metadata_df = pd.DataFrame(metadata)
        metadata_df["x"] = reduced_embeddings[:, 0]
        metadata_df["y"] = reduced_embeddings[:, 1]

        # Create Plotly visualization
        fig = px.scatter(
            metadata_df,
            x="x",
            y="y",
            color="language",
            hover_data=["dataset_name", "language", "id"],
            title=plot_title,
        )
        fig.show()
        logger.info("Visualization completed")
        if output_dir:
            fig.write_html(os.path.join(output_dir, f"{reduction_method}-{plot_title}-visualization.html"))

    @staticmethod
    def visualize_embeddings_2(embeddings, metadata, reduction_method="umap", plot_title="Embedding Visualization"):
    # Dimensionality reduction with UMAP
        if reduction_method == "umap":
            reducer = umap.UMAP(random_state=42, metric='cosine')
        elif reduction_method == "tsne":
            reducer = TSNE(random_state=42)
        else:
            logger.error("Invalid reduction method. Use 'umap' or 'tsne'.")
            raise ValueError("Invalid reduction method. Use 'umap' or 'tsne'.")
        reduced_embeddings = reducer.fit_transform(embeddings)

        metadata_df = pd.DataFrame(metadata)
        metadata_df["x"] = reduced_embeddings[:, 0]
        metadata_df["y"] = reduced_embeddings[:, 1]


        # Map tasks to colors and labels to shapes
        task_colors = {task: color for task, color in zip(metadata_df["dataset_name"].unique(), cc.glasbey_light)}
        label_shapes = {label: shape for label, shape in
                        zip(metadata_df["label"].unique(), ["circle", "square", "triangle", "cross", "hexagon"])}

        # Assign a unique shape for each label
        metadata_df["shape"] = metadata_df["label"].map(label_shapes)

        # Datashader Canvas
        canvas = ds.Canvas(plot_width=1000, plot_height=1000)

        # Aggregate data
        metadata_df['dataset_name'] = metadata_df['dataset_name'].astype('category')
        agg = canvas.points(metadata_df, "x", "y", ds.count_cat("dataset_name"))
        image = tf.shade(agg, color_key=task_colors, how="eq_hist")

        # Overlay shapes for labels
        fig, ax = plt.subplots(figsize=(10, 10))
        dsshow(agg, cmap=cc.glasbey_light, ax=ax)

        # Create a custom legend
        task_legend = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=task) for
                       task, color in task_colors.items()]
        shape_legend = [plt.Line2D([0], [0], marker=shape, color="k", markersize=10, label=label) for label, shape in
                        label_shapes.items()]
        plt.legend(handles=task_legend + shape_legend, loc="upper right", fontsize=10, title="Tasks & Labels")

        # Add titles
        plt.title("UMAP Visualization with Datashader", fontsize=14)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")

        fig.show()
        plt.show()

