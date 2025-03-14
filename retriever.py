import os
import random
from collections import Counter

import faiss
import torch
import numpy as np
from datasets import Dataset
import json
import logging

import math

from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer

from utils import convert_to_serializable

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
            index = faiss.IndexHNSWFlat(embedding_dim, 128)  # , faiss.METRIC_INNER_PRODUCT)
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

    def add_embeddings(self, embeddings, metadata, mmr_threshold=0.0):
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

        if mmr_threshold > 0:
            # Construct initial result list
            indices = [i for i in range(len(embeddings))]
            scores = [1 for _ in range(len(embeddings))]
            remained_indices = self.mmr_diversity_filter(embeddings=embeddings,  indices=indices, scores=scores,
                                                         similarity_threshold=mmr_threshold, batch_size=40000)
            embeddings = embeddings[remained_indices]
            metadata = [metadata[i] for i in remained_indices]

        # Add embeddings to the FAISS index
        self.index.add(embeddings)

        # Add metadata corresponding to embeddings
        self.metadata.extend(metadata)

        logger.info("Embeddings added successfully.")

    def retrieve_one_query(self, query_embedding, k=5, cluster_criteria_weight=0.0, unique_word_criteria_weight=0.0,
                           uncertainty_weight=0.0, margin_weight=0.0, perplexity_weight=0.0, balance_labels=False,
                           mmr_threshold=0.0):
        """
        Retrieve top-k nearest neighbors for a single query embedding, incorporating additional scoring weights.

        Args:
            query_embedding (np.ndarray): Query embedding vector (1 x embedding_dim).
            k (int): Number of nearest neighbors to retrieve.
            include_datasets (list[str]): Datasets to include from results.
            include_languages (list[str]): Languages to include from results.
            cluster_criteria_weight (float): Weight for clustering score in final ranking.
            unique_word_criteria_weight (float): Weight for unique word count in final ranking.
            uncertainty_weight (float): Weight for uncertainty in final ranking.
            margin_weight (float): Weight for margin in final ranking.
            perplexity_weight (float): Weight for perplexity in final ranking.
            balance_labels (bool): Whether to balance the retrieved results by label.
            mmr_threshold (float): Threshold for maximal marginal relevance.

        Returns:
            list[dict]: Top results based on combined scores, with optional filtering.
        """
        # logger.info("Performing retrieval for a single query.")

        # Ensure query_embedding is 2D for consistency with faiss interface
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize embeddings if required
        if self.normalize_index:
            faiss.normalize_L2(query_embedding)

        retrieval_number = k * 3
        results = []
        while len(results) < k * 3:
            # Perform search
            distances, indices = self.index.search(query_embedding, retrieval_number)

            # Filter out invalid indices (-1)
            valid_mask = indices[0] != -1
            distances, indices = distances[0][valid_mask], indices[0][valid_mask]

            # Fetch metadata for all valid indices
            metadata = [self.metadata[idx] for idx in indices]

            # Normalize distances
            norm_distances = self._min_max_scale(distances)

            # Construct initial result list
            results = [{"metadata": meta, "score": float(dist), "index": index}
                       for meta, dist, index in zip(metadata, norm_distances, indices)]

            # Deduplicate results
            results = self._deduplicate_results(results)
            # logger.info(f"Total unique results after deduplication: {len(results)}")

            # Apply MMR if threshold is provided
            if mmr_threshold > 0.0:
                results = self.mmr_wraper(results, similarity_threshold=mmr_threshold, min_remained_amount=k, lambda_param=0.5)

            retrieval_number *= 2

        # Compute additional feature scores
        norm_word_counts = self._compute_word_count_scores(results, unique_word_criteria_weight)
        norm_cluster_scores = self._compute_cluster_scores(results, cluster_criteria_weight)
        norm_uncertainty_scores = self._compute_normalized_scores(results, "uncertainty",
                                                                  uncertainty_weight, revert=True)
        norm_margin_scores = self._compute_normalized_scores(results, "margin", margin_weight)
        norm_perplexity_scores = self._compute_normalized_scores(results, "perplexity",
                                                                 perplexity_weight, revert=True)

        # Compute final scores with proper weighting
        results = self._compute_final_scores(
            results, norm_word_counts, norm_cluster_scores, norm_uncertainty_scores, norm_margin_scores,
            norm_perplexity_scores,
            cluster_criteria_weight, unique_word_criteria_weight, uncertainty_weight, margin_weight, perplexity_weight
        )

        # Sort results by final score
        results.sort(key=lambda x: x["score"])

        # Apply label balancing if requested
        if balance_labels:
            results = self._balance_labels(results, k)
        else:
            results = results[:k]

        # logger.info(f"Returning {len(results)} results.")

        return results

    def _retrieve_vectors(self, indices):
        """Reconstruct vectors from FAISS index based on non-consecutive indices."""
        valid_indices = [idx for idx in indices if idx != -1]  # Filter out invalid indices
        vectors = np.zeros((len(valid_indices), self.index.d), dtype=np.float32)  # Preallocate numpy array
        for i, idx in enumerate(valid_indices):
            vectors[i] = self.index.reconstruct(int(idx))
        return vectors

    def retrieve_multiple_queries(self, query_embeddings, k=5, num_retrieved=None, exclude_datasets=None,
                                  exclude_languages=None, cluster_criteria_weight=0.0, unique_word_criteria_weight=0.0,
                                  uncertainty_weight=0.0, margin_weight=0.0, perplexity_weight=0.0,
                                  balance_labels=False, mmr_threshold=0.0):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings, incorporating additional scoring weights.

        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            num_retrieved (int): Maximum number of results to return overall across all queries.
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

        results = []
        while len(results) < num_retrieved * 2:
            logger.info(f"--------------- K is: {k} -----------------")

            # Perform search for all queries
            distances, indices = self.index.search(query_embeddings, k)

            # Flatten distances and indices
            flattened_distances, flattened_indices = distances.flatten(), indices.flatten()

            # Filter out invalid indices (-1)
            valid_mask = flattened_indices != -1
            flattened_distances, flattened_indices = flattened_distances[valid_mask], flattened_indices[valid_mask]

            # Fetch metadata for all valid indices
            metadata = [self.metadata[idx] for idx in flattened_indices]
            logger.info(f"Total unique meta after searching: {len(metadata)}")

            # Apply filtering by language and dataset
            metadata, flattened_distances, flattened_indices = self._apply_exclude_filters(
                metadata, flattened_distances, flattened_indices, exclude_languages, exclude_datasets
            )
            logger.info(f"Total unique results after excluding: {len(metadata)}")

            # Normalize distances
            norm_distances = self._min_max_scale(flattened_distances)

            # Construct initial result list
            results = [{"metadata": meta, "score": float(dist), "index": index}
                       for meta, dist, index in zip(metadata, norm_distances, flattened_indices)]

            # Deduplicate results
            results = self._deduplicate_results(results)
            logger.info(f"Total unique results after deduplication: {len(results)}")

            k = k*3//2

        if mmr_threshold > 0:
            results = self.mmr_wraper(results, similarity_threshold=mmr_threshold, lambda_param=0.5,
                                                min_remained_amount=num_retrieved)

        # Compute additional feature scores
        norm_word_counts = self._compute_word_count_scores(results, unique_word_criteria_weight)
        norm_cluster_scores = self._compute_cluster_scores(results, cluster_criteria_weight)
        norm_uncertainty_scores = self._compute_normalized_scores(results, "uncertainty",
                                                                  uncertainty_weight, revert=True)
        norm_margin_scores = self._compute_normalized_scores(results, "margin", margin_weight)
        norm_perplexity_scores = self._compute_normalized_scores(results, "perplexity",
                                                                 perplexity_weight, revert=True)

        # Compute final scores with proper weighting
        results = self._compute_final_scores(
            results, norm_word_counts, norm_cluster_scores, norm_uncertainty_scores, norm_margin_scores,
            norm_perplexity_scores,
            cluster_criteria_weight, unique_word_criteria_weight, uncertainty_weight, margin_weight, perplexity_weight
        )

        # Sort results by final score
        results.sort(key=lambda x: x["score"])

        # Apply max retrieval constraint
        results = results[:num_retrieved] if not balance_labels else self._balance_labels(results, num_retrieved)

        logger.info(f"Returning {len(results)} results after applying num_retrieved limit.")

        return results

    # --------------- 🛠️ HELPER FUNCTIONS -----------------

    def _apply_exclude_filters(self, metadata, distances, indices, exclude_languages, exclude_datasets):
        """Apply language and dataset filters to metadata and distances."""
        if exclude_languages:
            metadata, distances, indices = self._filter_metadata(metadata, distances, indices, 'language',
                                                                 exclude_languages)
        if exclude_datasets:
            metadata, distances, indices = self._filter_metadata(metadata, distances, indices, 'dataset_name',
                                                                 exclude_datasets)
        return metadata, distances, indices

    def _apply_include_filters(self, metadata, distances, indices, include_languages, include_datasets):
        """Apply language and dataset filters to metadata and distances."""
        if include_languages:
            metadata, distances, indices = self._filter_metadata(metadata, distances, indices, 'language',
                                                                 include_languages)
        if include_datasets:
            metadata, distances, indices = self._filter_metadata(metadata, distances, indices, 'dataset_name',
                                                                 include_datasets)
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

        remained_indices = np.array([res["index"] for res in results])
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
                              norm_margin_scores, norm_perplexity_scores, cluster_criteria_weight,
                              unique_word_criteria_weight, uncertainty_weight, margin_weight, perplexity_weight):
        """Compute final weighted scores for retrieved results."""
        weight_sum = 1 - (
                cluster_criteria_weight + unique_word_criteria_weight + uncertainty_weight + margin_weight + perplexity_weight)

        return [
            {
                **res,
                "score": (weight_sum * res['score'] +
                          norm_word_counts[i] * unique_word_criteria_weight +
                          norm_cluster_scores[i] * cluster_criteria_weight +
                          norm_uncertainty_scores[i] * uncertainty_weight +
                          norm_margin_scores[i] * margin_weight +
                          norm_perplexity_scores[i] * perplexity_weight),
                "dist": res['score'],
                **({'length_score': norm_word_counts[i]} if unique_word_criteria_weight > 0 else {}),
                **({'cluster_score': norm_cluster_scores[i]} if cluster_criteria_weight > 0 else {}),
                **({'uncertainty_score': norm_uncertainty_scores[i]} if uncertainty_weight > 0 else {}),
                **({'margin_score': norm_margin_scores[i]} if margin_weight > 0 else {}),
                **({'perplexity_score': norm_perplexity_scores[i]} if perplexity_weight > 0 else {})
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
        cluster_sizes = {cluster: count for cluster, count in
                         cluster_counts.items()}  # Smaller clusters get higher scores

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

    def _filter_metadata(self, metadata, distances, indices, key, values, include_vs_exclude='exclude'):
        """Filter metadata based on exclude values."""
        mask = [meta.get(key) not in values for meta in metadata] \
            if include_vs_exclude == 'exclude' else [meta.get(key) in values for meta in metadata]
        filtered_metadata = [meta for meta, keep in zip(metadata, mask) if keep]
        filtered_distances = distances[mask]
        flattened_indices = indices[mask]
        return filtered_metadata, filtered_distances, flattened_indices

    def _deduplicate_results(self, results):
        """Deduplicate results by a unique key."""
        seen = set()
        deduplicated_results_1 = []
        for result in results:
            unique_key = result["metadata"].get("dataset_name") + result["metadata"].get("id")
            if unique_key not in seen:
                seen.add(unique_key)
                deduplicated_results_1.append(result)

        seen = set()
        deduplicated_results_2 = []
        for result in deduplicated_results_1:
            unique_text = result["metadata"].get("text")
            if unique_text not in seen:
                seen.add(unique_text)
                deduplicated_results_2.append(result)

        return deduplicated_results_2

    def retrieve_random_metadata(self, num_retrieved=None, exclude_datasets=None, exclude_languages=None,
                                 cluster_criteria_weight=0.0, unique_word_criteria_weight=0.0,
                                 uncertainty_weight=0.0, margin_weight=0.0, perplexity_weight=0.0,
                                 balance_labels=False, mmr_threshold=0.0):
        """
        Retrieve a random selection of metadata, with optional filtering and criteria-based scoring.

        Args:
            num_retrieved (int): Number of metadata entries to retrieve after applying criteria.
            exclude_datasets (list[str]): List of dataset names to exclude from results.
            exclude_languages (list[str]): List of languages to exclude from results.
            cluster_criteria_weight (float): Weight for clustering score in final ranking.
            unique_word_criteria_weight (float): Weight for unique word count in final ranking.
            uncertainty_weight (float): Weight for uncertainty in final ranking.
            margin_weight (float): Weight for margin in final ranking.
            perplexity_weight (float): Weight for perplexity in final ranking.
            balance_labels (bool): Whether to balance the retrieved results by label.
            mmr_threshold (float): Threshold for maximal marginal relevance.

        Returns:
            list[dict]: Randomly selected metadata entries after applying optional filters and scoring criteria.
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

        # Sample more entries than needed to allow for criteria-based filtering and ranking
        sample_size = max(num_retrieved * 2, 100)

        # Get random indices for selected metadata
        indices = list(range(len(filtered_metadata)))
        random_indices = random.sample(indices, sample_size)

        # Create initial results with metadata and indices
        results = []
        for i, idx in enumerate(random_indices):
            meta = filtered_metadata[idx]
            # Use random scores as initial values
            results.append({
                "metadata": meta,
                "score": 0,  # Random initial score
                "index": idx
            })

        # Deduplicate results based on content
        results = self._deduplicate_results(results)
        logger.info(f"Total unique results after deduplication: {len(results)}")

        # Apply MMR if threshold is provided
        if mmr_threshold > 0.0:
            results = self.mmr_wraper(results, similarity_threshold=mmr_threshold, lambda_param=0.5,
                                      min_remained_amount=num_retrieved)

        # Compute additional feature scores
        norm_word_counts = self._compute_word_count_scores(results, unique_word_criteria_weight)
        norm_cluster_scores = self._compute_cluster_scores(results, cluster_criteria_weight)
        norm_uncertainty_scores = self._compute_normalized_scores(results, "uncertainty",
                                                                  uncertainty_weight, revert=True)
        norm_margin_scores = self._compute_normalized_scores(results, "margin", margin_weight)
        norm_perplexity_scores = self._compute_normalized_scores(results, "perplexity",
                                                                 perplexity_weight, revert=True)

        # Compute final scores with proper weighting
        results = self._compute_final_scores(
            results, norm_word_counts, norm_cluster_scores, norm_uncertainty_scores, norm_margin_scores,
            norm_perplexity_scores,
            cluster_criteria_weight, unique_word_criteria_weight, uncertainty_weight, margin_weight, perplexity_weight
        )

        # Sort results by final score
        results.sort(key=lambda x: x["score"])

        # Apply number of results limit with optional label balancing
        if balance_labels:
            results = self._balance_labels(results, num_retrieved)
        else:
            results = results[:num_retrieved]

        logger.info(f"Returning {len(results)} randomly selected entries after applying criteria.")

        return results

    def save_meta_to_file(self, meta, path):
        """
        Saves metadata as a JSON file, ignoring non-serializable entries.

        Args:
            meta (dict or list): Metadata to save.
            path (str): Directory where the file will be saved.
        """

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


    def mmr_wraper(self, results, similarity_threshold=0.95, min_remained_amount=None, lambda_param=0.5, batch_size=20000):


        # Check input data
        indices = np.array([res["index"] for res in results])
        scores = self._min_max_scale(np.array([-1 * res["score"] for res in results]), )
        embeddings = self._retrieve_vectors(indices)

        selected_idx_original = self.mmr_diversity_filter(embeddings=embeddings, indices=indices, scores=scores,
                                                           similarity_threshold=similarity_threshold,
                                                           min_remained_amount=min_remained_amount,
                                                           lambda_param=lambda_param, batch_size=batch_size)


        selected_results = [res for res in results if res['index'] in selected_idx_original]
        return selected_results


    def mmr_diversity_filter(self, embeddings, indices, scores, similarity_threshold=0.95, lambda_param=0.5,
                             min_remained_amount=None, batch_size=20000):
        """
        Apply Maximal Marginal Relevance (MMR) to filter similar embeddings while ensuring
        at least min_remained_amount embeddings are kept.

        Args:
            results: List of dictionaries containing indices and scores
            similarity_threshold: Similarity threshold above which embeddings are considered too similar
            lambda_param: Trade-off between relevance and diversity (0 to 1)
            min_remained_amount: Minimum number of embeddings to keep regardless of similarity
            batch_size: Size of batches for processing large datasets

        Returns:
            List of dictionaries for the diverse subset of results
        """
        if similarity_threshold == 0.0 or len(embeddings) == 0:
            return indices

        # Ensure min_remained_amount is valid
        if min_remained_amount is None:
            min_remained_amount = max(1, int(len(embeddings) * 0.1))  # Default to 10% of data
        min_remained_amount = min(max(1, min_remained_amount), len(embeddings))
        logger.info(f"Minimum embeddings to keep: {min_remained_amount}")
        embeddings_tensor = None
        # Process in batches to prevent OOM
        try:
            with torch.no_grad():  # Disable gradient computation
                # Convert to PyTorch tensors and normalize
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                embeddings_norm = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
                embeddings_norm = embeddings_norm.to(self.device)
                scores = torch.tensor(scores, dtype=torch.float32).to(self.device)

                # Initialize with the first embedding
                selected_indices = [0]
                selected_idx_original = [indices[0]]

                # Process remaining embeddings
                remaining_indices = list(range(1, len(embeddings)))

                # Keep track of selected embeddings (but keep on CPU when not in use)
                selected_embeddings_cpu = embeddings_norm[selected_indices].clone().cpu()

                while remaining_indices:
                    # Calculate scores in batches
                    max_score = float('-inf')
                    max_idx = -1

                    for batch_start in range(0, len(remaining_indices), batch_size):
                        batch_indices = remaining_indices[batch_start:batch_start + batch_size]
                        batch_embeddings = embeddings_norm[batch_indices]

                        # Move selected embeddings to GPU for this computation
                        selected_embeddings = selected_embeddings_cpu.to(self.device)

                        # Relevance term
                        relevance_scores = scores[batch_indices]

                        # Diversity term: Max similarity to already selected embeddings
                        similarity_matrix = torch.matmul(batch_embeddings, selected_embeddings.T)
                        max_similarities, _ = torch.max(similarity_matrix, dim=1)

                        # MMR score: balance between relevance and diversity
                        batch_scores = lambda_param * relevance_scores - (1 - lambda_param) * max_similarities

                        # Find best candidate in batch
                        batch_max_val, batch_max_idx = torch.max(batch_scores, dim=0)
                        batch_max_score = batch_max_val.item()
                        batch_max_index = batch_max_idx.item()

                        if batch_max_score > max_score:
                            max_score = batch_max_score
                            max_idx = batch_indices[batch_max_index]

                        # Clear GPU memory for this batch
                        del similarity_matrix, max_similarities, batch_scores
                        selected_embeddings = selected_embeddings.cpu()  # Move back to CPU

                        # Force CUDA memory cleanup
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()

                    # Check if best candidate is too similar to any selected embedding
                    candidate_embedding = embeddings_norm[max_idx].unsqueeze(0)
                    selected_embeddings = selected_embeddings_cpu.to(self.device)
                    similarity_values = torch.matmul(candidate_embedding, selected_embeddings.T)
                    max_similarity = torch.max(similarity_values).item()

                    # Only stop if we have enough embeddings AND the next best is too similar
                    if len(selected_indices) >= min_remained_amount and max_similarity > similarity_threshold:
                        logger.info(
                            f"Stopping: reached {len(selected_indices)} embeddings with next similarity {max_similarity:.4f}")
                        break

                    # Add the embedding with the highest score
                    selected_indices.append(max_idx)
                    selected_idx_original.append(indices[max_idx])
                    remaining_indices.remove(max_idx)

                    # Update selected embeddings tensor (on CPU)
                    selected_embeddings_cpu = torch.cat([selected_embeddings_cpu, candidate_embedding.cpu()], dim=0)

                    # Clean up GPU
                    del similarity_values, candidate_embedding
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

                    # Log progress at intervals
                    if len(selected_indices) % 1000 == 0:
                        logger.info(f"Selected {len(selected_indices)}/{len(embeddings)} embeddings")

        finally:
            # Final cleanup - delete all tensors and clear cache
            if self.device == 'cuda' and embeddings_tensor is not None:
                del embeddings_tensor, embeddings_norm, scores
                if 'selected_embeddings' in locals():
                    del selected_embeddings
                torch.cuda.empty_cache()

        # Final logging
        logger.info(f"Final selection: {len(selected_indices)} diverse embeddings out of {len(embeddings)}")
        logger.info(f"Removed {len(embeddings) - len(selected_indices)} similar embeddings")

        return selected_idx_original
