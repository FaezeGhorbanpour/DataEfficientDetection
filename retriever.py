import os
import random
from collections import Counter

import faiss
import numpy as np
from datasets import Dataset
import json
import logging

from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedding_dim=768, index_type="FlatL2", device="cuda", normalize_index=False):
        """
        Initialize the Retriever with an FAISS index.
        Args:
            embedding_dim (int): Dimension of the embeddings.
            index_type (str): FAISS index type (e.g., "FlatL2", "IVF", "HNSW").
            device (str): Device to use ("cuda" or "cpu").
        """
        logger.info("Initializing the retriever module...")
        self.device = device
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.normalize_index = normalize_index

        self.index = self._initialize_index(embedding_dim, index_type)
        self.metadata = []
        logger.info(f"Retriever initialized with {index_type} index on {device}")

    def _initialize_index(self, embedding_dim, index_type):
        """
        Initialize FAISS index based on the given type.
        Args:
            embedding_dim (int): Dimension of embeddings.
            index_type (str): Type of FAISS index.
        Returns:
            faiss.Index: Initialized FAISS index.
        """
        logger.info(f"Creating index of type: {index_type}")
        if index_type == "FlatL2":
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "FlatIP":
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(embedding_dim, 128)#, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200  # Set parameter for construction
            index.hnsw.efSearch = 128  # Set parameter for search
        elif index_type == "IVF":
            index = faiss.IndexIVFPQ(faiss.IndexFlatL2(embedding_dim), 100, 32,
                                     32)  # Number of centroids and PQ settings
        else:
            logger.error(f"Unknown index_type: {index_type}")
            raise ValueError(f"Unknown index_type: {index_type}")

        # Move the index to GPU if needed
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS index.")
            return faiss.index_cpu_to_all_gpus(index)

        logger.info("Using CPU for FAISS index.")
        return index

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and their corresponding metadata to the index.
        Args:
            embeddings (np.ndarray): Embedding vectors to add (N x embedding_dim).
            metadata (list[dict]): Metadata associated with each embedding.
        """
        logger.info(f"Adding {len(embeddings)} embeddings to the index.")

        # Normalize embeddings if required
        if self.normalize_index:
            faiss.normalize_L2(embeddings)

        # Add embeddings to the FAISS index
        self.index.add(embeddings)

        # Add metadata corresponding to embeddings
        self.metadata.extend(metadata)

        logger.info("Embeddings added successfully.")

    def retrieve(self, query_embedding, k=5, filters=None):
        """
        Retrieve top-k nearest neighbors for a given query embedding, optionally filtering by metadata.
        Args:
            query_embedding (np.ndarray): Query embedding vector (1 x embedding_dim).
            k (int): Number of nearest neighbors to retrieve.
            filters (dict): Metadata filters (e.g., {"language": "en", "dataset_name": "my_dataset"}).
        Returns:
            list[dict]: Retrieved metadata and scores.
        """
        logger.info("Performing retrieval for a single query.")

        # Normalize embeddings if required
        if self.normalize_index:
            faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # No more valid indices
                continue
            data_meta = self.metadata[idx]
            if filters:
                if not all(data_meta.get(key) == value for key, value in filters.items()):
                    continue
            results.append({"metadata": data_meta, "score": dist})
        logger.info(f"Retrieved {len(results)} results.")
        return results

    def _retrieve_vectors(self, indices):
        """Reconstruct vectors from FAISS index based on non-consecutive indices."""
        valid_indices = [idx for idx in indices if idx != -1]  # Filter out invalid indices
        vectors = np.zeros((len(valid_indices), self.index.d), dtype=np.float32)  # Preallocate numpy array
        for i, idx in enumerate(valid_indices):
            vectors[i] = self.index.reconstruct(int(idx))
        return vectors


    def retrieve_multiple_queries(self, query_embeddings, k=5, max_retrieved=None, exclude_datasets=None,
                                  exclude_languages=None, cluster_criteria_weight=0.0, unique_word_criteria_weight=0.0,
                                  uncertainty_weight=0.0, perplexity_weight=0.0, balance_labels=False):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings, incorporating additional scoring weights.

        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            max_retrieved (int): Maximum number of results to return overall across all queries.
            exclude_datasets (list[str]): Datasets to exclude from results.
            exclude_languages (list[str]): Languages to exclude from results.
            cluster_criteria_weight (float): Weight for clustering score in final ranking.
            unique_word_criteria_weight (float): Weight for unique word count in final ranking.
            uncertainty_weight (float): Weight for uncertainty in final ranking.
            perplexity_weight (float): Weight for perplexity in final ranking.
            balance_labels (bool): Whether to balance the retrieved results by label.

        Returns:
            list[dict]: Top results based on combined scores, with optional filtering and deduplication.
        """
        logger.info("Starting retrieval for multiple queries.")

        # Normalize query embeddings if required
        if self.normalize_index:
            faiss.normalize_L2(query_embeddings)

        # Perform search for all queries
        distances, indices = self.index.search(query_embeddings, k)

        # Flatten distances and indices
        flattened_distances, flattened_indices = distances.flatten(), indices.flatten()

        # Filter out invalid indices (-1)
        valid_mask = flattened_indices != -1
        flattened_distances, flattened_indices = flattened_distances[valid_mask], flattened_indices[valid_mask]

        # Fetch metadata for all valid indices
        metadata = [self.metadata[idx] for idx in flattened_indices]

        # Apply filtering by language and dataset
        metadata, flattened_distances, flattened_indices = self._apply_filters(
            metadata, flattened_distances, flattened_indices, exclude_languages, exclude_datasets
        )

        # Normalize distances
        norm_distances = self._min_max_scale(flattened_distances)

        # Construct initial result list
        results = [{"metadata": meta, "score": float(dist), "index": index}
                   for meta, dist, index in zip(metadata, norm_distances, flattened_indices)]

        # Deduplicate results
        results = self._deduplicate_results(results)
        logger.info(f"Total unique results after deduplication: {len(results)}")

        # Compute additional feature scores
        norm_word_counts = self._compute_word_count_scores(results, unique_word_criteria_weight)
        norm_cluster_scores = self._compute_cluster_scores(results, cluster_criteria_weight)
        norm_uncertainty_scores = self._compute_normalized_scores(results, "margin",
                                                                  uncertainty_weight, revert=True)
        norm_perplexity_scores = self._compute_normalized_scores(results, "perplexity",
                                                                 perplexity_weight, revert=True)

        # Compute final scores with proper weighting
        results = self._compute_final_scores(
            results, norm_word_counts, norm_cluster_scores, norm_uncertainty_scores, norm_perplexity_scores,
            cluster_criteria_weight, unique_word_criteria_weight, uncertainty_weight, perplexity_weight
        )

        # Sort results by final score
        results.sort(key=lambda x: x["score"])

        # Apply max retrieval constraint
        results = results[:max_retrieved] if not balance_labels else self._balance_labels(results, max_retrieved)

        logger.info(f"Returning {len(results)} results after applying max_retrieved limit.")

        return results

    # --------------- 🛠️ HELPER FUNCTIONS -----------------

    def _apply_filters(self, metadata, distances, indices, exclude_languages, exclude_datasets):
        """Apply language and dataset filters to metadata and distances."""
        if exclude_languages:
            metadata, distances, indices = self._filter_metadata(metadata, distances, indices, 'language',
                                                                 exclude_languages)
        if exclude_datasets:
            metadata, distances, indices = self._filter_metadata(metadata, distances, indices, 'dataset_name',
                                                                 exclude_datasets)
        return metadata, distances, indices

    def _compute_word_count_scores(self, results, unique_word_criteria_weight):
        """Compute word count scores if the weight is provided."""
        if unique_word_criteria_weight == 0:
            return np.zeros(len(results), dtype=np.float32)

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
        token_counts = np.array([-1 * len(set(tokenizer.tokenize(res['metadata']["text"]))) for res in results])
        return self._min_max_scale(token_counts)

    def _compute_cluster_scores(self, results, cluster_criteria_weight):
        """Compute clustering scores if the weight is provided."""
        if cluster_criteria_weight == 0:
            return np.zeros(len(results), dtype=np.float32)

        remained_indices = [res["index"] for res in results]
        all_embeddings = self._retrieve_vectors(remained_indices)
        num_clusters = int(max(len(all_embeddings) / 100, 50))
        cluster_scores = self.cluster_scores(all_embeddings, remained_indices,
                                                      num_clusters=num_clusters)
        return self._min_max_scale(cluster_scores)

    def _compute_normalized_scores(self, results, metric_name, weight, revert=False):
        """Retrieve and normalize metric-based scores (uncertainty, perplexity) if the weight is provided."""
        if weight == 0:
            return np.zeros(len(results), dtype=np.float32)
        if revert:
            scores = np.array([-1 * res["metadata"].get(metric_name, 0) for res in results])
        else:
            scores = np.array([res["metadata"].get(metric_name, 0) for res in results])
        return self._min_max_scale(scores)

    def _compute_final_scores(self, results, norm_word_counts, norm_cluster_scores, norm_uncertainty_scores,
                              norm_perplexity_scores,
                              cluster_criteria_weight, unique_word_criteria_weight, uncertainty_weight,
                              perplexity_weight):
        """Compute final weighted scores for retrieved results."""
        weight_sum = 1 - (
                    cluster_criteria_weight + unique_word_criteria_weight + uncertainty_weight + perplexity_weight)

        return [
            {**res,
             "score": (weight_sum * res['score'] +
                       norm_word_counts[i] * unique_word_criteria_weight +
                       norm_cluster_scores[i] * cluster_criteria_weight +
                       norm_uncertainty_scores[i] * uncertainty_weight +
                       norm_perplexity_scores[i] * perplexity_weight),
             "dist": res['score'],
             "length_score": norm_word_counts[i],
             "cluster_score": norm_cluster_scores[i],
             "uncertainty_score": norm_uncertainty_scores[i],
             "perplexity_score": norm_perplexity_scores[i]
             }
            for i, res in enumerate(results)
        ]

    # def _compute_unique_word_count(self, text):
    #     """Compute the number of unique words in a text."""
    #     return 1.0/len(set(text.lower().split()))
    #

    def cluster_scores(self, embeddings, indices, num_clusters):
        """Cluster embeddings and compute cluster scores based on cluster sizes."""
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=256, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_counts = Counter(cluster_labels)
        cluster_sizes = {cluster: count for cluster, count in cluster_counts.items()}  # Smaller clusters get higher scores

        return np.array([cluster_sizes[cluster_labels[i]] for i, idx in enumerate(indices)])

    def _min_max_scale(self, arr):
        """Apply Min-Max normalization to scale values between 0 and 1."""
        if len(arr) == 0:
            return arr
        min_val, max_val = np.min(arr), np.max(arr)
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(arr)

    def _balance_labels(self, results, max_count):
        """Ensure the results contain a balanced number of label 0 and label 1 samples, preserving order."""
        # Efficiently separate the results into label 0 and label 1 using numpy arrays for better speed
        labels = np.array([r["metadata"]["label"] for r in results])

        # Find indices where the labels are 0 or 1
        label_0_indices = np.where(labels == 0)[0]
        label_1_indices = np.where(labels == 1)[0]

        # Calculate the minimum count for balancing
        min_count = min(len(label_0_indices), len(label_1_indices), max_count // 2)

        # Select the first `min_count` entries from each group, preserving their order
        balanced_indices = np.concatenate([label_0_indices[:min_count], label_1_indices[:min_count]])

        # Sort the balanced indices to preserve the order in `results`
        balanced_indices.sort()

        # Efficiently fetch the balanced results using the selected and sorted indices
        return [results[i] for i in balanced_indices]


    def _filter_metadata(self, metadata, distances, indices, key, exclude_values):
        """Filter metadata based on exclude values."""
        mask = [meta.get(key) not in exclude_values for meta in metadata]
        filtered_metadata = [meta for meta, keep in zip(metadata, mask) if keep]
        filtered_distances = distances[mask]
        flattened_indices = indices[mask]
        return filtered_metadata, filtered_distances, flattened_indices

    def _deduplicate_results(self, results):
        """Deduplicate results by a unique key."""
        seen = set()
        deduplicated_results = []
        for result in results:
            unique_key = result["metadata"].get("dataset_name") + result["metadata"].get("id")
            if unique_key not in seen:
                seen.add(unique_key)
                deduplicated_results.append(result)
        return deduplicated_results

    def retrieve_random_metadata(self, max_retrieved=None, exclude_datasets=None, exclude_languages=None,
                                 cluster_criteria_weight=0.0, unique_word_criteria_weight=0.0, balance_labels=False):
        """
        Retrieve a random selection of metadata, with optional filtering.

        Args:
            max_retrieved (int): Number of random metadata entries to retrieve.
            exclude_datasets (list[str]): List of dataset names to exclude from results.
            exclude_languages (list[str]): List of languages to exclude from results.

        Returns:
            list[dict]: Randomly selected metadata entries after applying optional filters.
        """
        logger.info("Starting random metadata retrieval.")

        # Filter metadata based on exclusion criteria
        filtered_metadata = self.metadata

        if exclude_languages:
            filtered_metadata = [
                meta for meta in filtered_metadata
                if meta.get('language') not in exclude_languages
            ]

        if exclude_datasets:
            filtered_metadata = [
                meta for meta in filtered_metadata
                if meta.get('dataset_name') not in exclude_datasets
            ]

        # Randomly sample the desired number of results
        num_results = min(max_retrieved, len(filtered_metadata))
        metadata = random.sample(filtered_metadata, num_results)
        # Combine distances and metadata into a single structure
        random_metadata = [{"metadata": meta} for meta in metadata]

        logger.info(f"Returning {len(random_metadata)} random metadata entries.")

        return random_metadata

    def save_meta_to_file(self, meta, path):
        """
        Saves metadata as a JSON file, ignoring non-serializable entries.

        Args:
            meta (dict or list): Metadata to save.
            path (str): Directory where the file will be saved.
        """

        def convert_to_serializable(obj):
            """Convert numpy types and ignore non-serializable objects."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):  # Convert numpy int
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):  # Convert numpy float
                return float(obj)
            elif isinstance(obj, np.ndarray):  # Convert numpy array
                return obj.tolist()
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple, set)):  # Convert lists/sets/tuples recursively
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):  # Convert dict recursively
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return None  # Ignore non-serializable objects

        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "retrieved_data.json")
            logger.info(f"Saving metadata at {file_path}")

            meta_serializable = convert_to_serializable(meta)

            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(meta_serializable, json_file, indent=4, ensure_ascii=False)

            logger.info(f"Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")


    def save_index(self, path):
        """
        Save the FAISS index and metadata to a file.
        Args:
            path (str): Directory to save the index and metadata.
        """
        logger.info(f"Saving index to {path}")
        if not os.path.exists(path):
            os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, 'embedding.index'))
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
        logger.info("Index saved successfully is the directory:")

    def load_index(self, path):
        """
        Load the FAISS index and metadata from a file.
        Args:
            path (str): Directory containing the index and metadata.
        """
        logger.info(f"Loading index from {path}")
        self.index = faiss.read_index(os.path.join(path, 'embedding.index'))
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        logger.info("Index loaded successfully.")
