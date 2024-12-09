

from FlagEmbedding import BGEM3FlagModel

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import pandas as pd
import plotly.express as px

class Embedder:
    def __init__(self, model_name):
        """
        Initialize the Embedder with a multilingual model.
        Args:
            model_name (str): Name of the Hugging Face model to load.
        """
        self.name = model_name
        if self.name == 'labse':
            self.model = SentenceTransformer('sentence-transformers/LaBSE')
        elif self.name.lower() == 'minilm':
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        elif self.name == 'm3':
            self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        elif self.name.lower() == 'e5-large':
            self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        elif self.name.lower() == 'e5-base':
            self.model = SentenceTransformer('intfloat/multilingual-e5-base')
        elif self.name.lower() == 'xlmr-large':
            self.model = SentenceTransformer('FacebookAI/xlm-roberta-large')
        elif self.name.lower() == 'xlmr-base':
            self.model = SentenceTransformer('FacebookAI/xlm-roberta-base')
        elif self.name.lower() == 'arctic-large':
            self.model = SentenceTransformer('Snowflake/snowflake-arctic-embed-l-v2.0')
        elif self.name.lower() == 'arctic-base':
            self.model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0')
        else:
            self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_sentences(self, sentences):
        """
        Embed a list of sentences.
        Args:
            sentences (list[str]): Sentences to embed.
        Returns:
            np.ndarray: Embedding vectors.
        """
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings.cpu().detach().numpy()

    def embed_datasets(self, datasets):
        """
        Embed all instances in multiple datasets.
        Args:
            datasets (list[dict]): Datasets with text to embed.
        Returns:
            np.ndarray, list[dict]: Embeddings and corresponding metadata.
        """
        embeddings = []
        metadata = []
        for dataset in datasets:
            texts = dataset["data"]["text"]
            labels = dataset["data"]["label"]
            ids = dataset["data"]["id"]
            embs = self.embed_sentences(texts)
            embeddings.append(embs)
            metadata += [{"text": texts[i],
                          "label": labels[i],
                          "id": ids[i],
                          "dataset_name": dataset["name"],
                          "language": dataset["language"]} for i in range(len(embs))]
        return embeddings, metadata

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        """
        norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)

        normalized1 = embedding1 / norm1
        normalized2 = embedding2 / norm2

        similarity_matrix = np.dot(normalized1, normalized2.T).diagonal()
        return np.mean(similarity_matrix)

    # @staticmethod
    # def cluster_embeddings(embeddings, num_clusters=5):
    #     """
    #     Cluster embeddings using KMeans.
    #     """
    #     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #     return kmeans.fit_predict(embeddings)

    # @staticmethod
    # def plot_embeddings(embeddings, metadata, title="UMAP Projection"):
    #     """
    #     Plot embeddings with UMAP, colored by language and dataset name.
    #     """
    #     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    #     reduced_embs = reducer.fit_transform(embeddings)
    #
    #     languages = [meta.get("language", "unknown") for meta in metadata]
    #     dataset_names = [meta["dataset_name"] for meta in metadata]
    #     plt.figure(figsize=(12, 8))
    #     for lang in set(languages):
    #         indices = [i for i, l in enumerate(languages) if l == lang]
    #         plt.scatter(reduced_embs[indices, 0], reduced_embs[indices, 1], label=lang)
    #     plt.title(title)
    #     plt.legend()
    #     plt.show()

    @staticmethod
    def cluster_embeddings(self, embeddings, metadata, n_clusters):
        """
        Cluster embeddings and return clusters with metadata.
        Args:
            embeddings (np.ndarray): Embeddings to cluster.
            metadata (list[dict]): Metadata for each embedding.
            n_clusters (int): Number of clusters.
        Returns:
            pd.DataFrame: Clustered data with metadata.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Create DataFrame with clusters and metadata
        data_with_clusters = pd.DataFrame(metadata)
        data_with_clusters["cluster"] = cluster_labels
        return data_with_clusters

    @staticmethod
    def visualize_embeddings(embeddings, metadata, reduction_method="umap", plot_title="Embedding Visualization"):
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
        if reduction_method == "umap":
            reducer = umap.UMAP(random_state=42)
        elif reduction_method == "tsne":
            reducer = TSNE(random_state=42, n_iter=300, perplexity=30)
        else:
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
            color="language",  # Adjust this to a relevant metadata field
            hover_data=metadata,
            title=plot_title,
        )
        fig.show()
