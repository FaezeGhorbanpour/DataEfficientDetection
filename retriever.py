import os
import random
from collections import Counter

import faiss
import torch
import numpy as np
import json
import logging

from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoTokenizer
''
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
                                                         similarity_threshold=mmr_threshold, batch_size=20000)
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
                                  balance_labels=False, mmr_threshold=0.0, lambda_param=0.5):
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
        num_search = 0
        while len(results) < num_retrieved * 2 and num_search < 10:
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

            k = k * 2
            num_search += 1

        if mmr_threshold > 0:
            results = self.mmr_wraper(results, similarity_threshold=mmr_threshold, lambda_param=lambda_param,
                                                min_remained_amount=num_retrieved)

            logger.info(f"Total results after MMR: {len(results)} with mmr threshold: {mmr_threshold}")

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


    def mmr_wraper(self, results, similarity_threshold=0.95, min_remained_amount=None, lambda_param=0.5):


        # Check input data
        indices = np.array([res["index"] for res in results])
        scores = self._min_max_scale(np.array([-1 * res["score"] for res in results]), )
        embeddings = self._retrieve_vectors(indices)

        selected_idx_original = self.mmr_diversity_filter_fast(embeddings=embeddings, indices=indices, scores=scores,
                                                           similarity_threshold=similarity_threshold,
                                                           min_remained_amount=min_remained_amount,
                                                           lambda_param=lambda_param, device=self.device
                                                            )


        selected_results = [res for res in results if res['index'] in selected_idx_original]
        return selected_results

    def mmr_diversity_filter(self, embeddings, indices, scores, similarity_threshold=0.95, lambda_param=0.5,
                             min_remained_amount=None, batch_size=20000, device='cuda'):
        """
        Optimized Maximal Marginal Relevance (MMR) implementation for large datasets
        using batch processing and multi-GPU support.

        Args:
            embeddings: List or array of embeddings
            indices: Original indices of the embeddings
            scores: Relevance scores for each embedding
            similarity_threshold: Similarity threshold above which embeddings are considered too similar
            lambda_param: Trade-off between relevance and diversity (0 to 1)
            min_remained_amount: Minimum number of embeddings to keep
            batch_size: Size of batches for processing
            max_workers: Number of GPUs to use (if available)

        Returns:
            List of original indices for the diverse subset of results
        """
        if similarity_threshold == 0.0 or len(embeddings) == 0:
            return indices

        # Ensure min_remained_amount is valid
        if min_remained_amount is None:
            min_remained_amount = max(1, int(len(embeddings) * 0.1))  # Default to 10% of data

        min_remained_amount = min(max(1, min_remained_amount), len(embeddings))
        logger.info(f"Minimum embeddings to keep: {min_remained_amount}")

        try:
            with torch.no_grad():  # Disable gradient computation
                # Convert to PyTorch tensors and normalize (keep on CPU initially)
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                norms = torch.norm(embeddings_tensor, dim=1, keepdim=True)
                embeddings_norm = embeddings_tensor / norms
                scores_tensor = torch.tensor(scores, dtype=torch.float32)

                # Get the index of highest scoring embedding
                first_idx = torch.argmax(scores_tensor).item()

                # Initialize with the highest scoring embedding
                selected_indices = [first_idx]
                selected_idx_original = [indices[first_idx]]

                # Create mask for remaining embeddings
                mask = torch.ones(len(embeddings), dtype=torch.bool)
                mask[first_idx] = False
                remaining_count = len(embeddings) - 1

                # Move selected embedding to device
                selected_embeddings = embeddings_norm[selected_indices].to(device)

                # Main loop
                pbar = tqdm(total=min(len(embeddings), min_remained_amount * 2))
                pbar.update(1)

                while remaining_count > 0 and len(selected_indices) < len(embeddings):
                    # Process in manageable batches
                    max_score = float('-inf')
                    max_idx = -1

                    # Get remaining indices
                    remaining_indices = torch.where(mask)[0]

                    for start_idx in range(0, remaining_count, batch_size):
                        end_idx = min(start_idx + batch_size, remaining_count)
                        batch_indices = remaining_indices[start_idx:end_idx]

                        # Move batch to device
                        batch_embeddings = embeddings_norm[batch_indices].to(device)
                        batch_scores = scores_tensor[batch_indices].to(device)

                        # Calculate similarity matrix for the batch against selected embeddings
                        similarity_matrix = torch.mm(batch_embeddings, selected_embeddings.T)

                        # Get maximum similarity for each embedding in batch to any selected embedding
                        max_similarities, _ = torch.max(similarity_matrix, dim=1)

                        # Calculate MMR scores
                        mmr_scores = lambda_param * batch_scores - (1 - lambda_param) * max_similarities

                        # Find best candidate in batch
                        batch_max_val, batch_max_idx = torch.max(mmr_scores, dim=0)
                        batch_max_score = batch_max_val.item()

                        if batch_max_score > max_score:
                            max_score = batch_max_score
                            max_idx = batch_indices[batch_max_idx].item()

                        # Clean up GPU memory
                        del batch_embeddings, batch_scores, similarity_matrix, max_similarities, mmr_scores
                        if device == 'cuda':
                            torch.cuda.empty_cache()

                    # Check if best candidate is too similar to any selected embedding
                    candidate_embedding = embeddings_norm[max_idx].unsqueeze(0).to(device)
                    similarity_values = torch.mm(candidate_embedding, selected_embeddings.T)
                    max_similarity = torch.max(similarity_values).item()

                    # Only stop if we have enough embeddings AND the next best is too similar
                    if len(selected_indices) >= min_remained_amount and max_similarity > similarity_threshold:
                        logger.info(
                            f"Stopping: reached {len(selected_indices)} embeddings with next similarity {max_similarity:.4f}")
                        break

                    # Add the embedding with the highest score
                    selected_indices.append(max_idx)
                    selected_idx_original.append(indices[max_idx])
                    mask[max_idx] = False
                    remaining_count -= 1

                    # Update selected embeddings tensor
                    selected_embeddings = torch.cat([selected_embeddings, candidate_embedding], dim=0)

                    # Clean up
                    del candidate_embedding, similarity_values
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                    # Update progress bar
                    pbar.update(1)


                pbar.close()

        except Exception as e:
            logger.error(f"Error in MMR filtering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # In case of error, return original indices
            return indices
        finally:
            # Final cleanup
            if device == 'cuda':
                torch.cuda.empty_cache()

        # Final logging
        logger.info(f"Final selection: {len(selected_indices)} diverse embeddings out of {len(embeddings)}")
        logger.info(f"Removed {len(embeddings) - len(selected_indices)} similar embeddings")

        return selected_idx_original


    def mmr_diversity_filter_fast(self, embeddings, indices, scores, similarity_threshold=0.99,
                                  lambda_param=0.5, min_remained_amount=None, device='cuda'):
        if similarity_threshold == 0.0 or len(embeddings) == 0:
            return indices

        if min_remained_amount is None:
            min_remained_amount = max(1, int(len(embeddings) * 0.75))

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float16).to(device)
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, dim=1)
        scores_tensor = torch.tensor(scores, dtype=torch.float16).to(device)

        N = len(embeddings)
        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
        selected_indices = []

        # Precompute full similarity matrix
        similarity_matrix = embeddings_tensor @ embeddings_tensor.T
        similarity_matrix.fill_diagonal_(-float('inf'))

        # Select the highest scoring embedding first
        current_idx = torch.argmax(scores_tensor).item()
        selected_indices.append(current_idx)
        selected_mask[current_idx] = True

        # Initialize max similarities to the already-selected embedding
        max_similarities = similarity_matrix[current_idx].clone()

        pbar = tqdm(total=N)
        pbar.update(1)

        for _ in range(N - 1):
            mmr_scores = lambda_param * scores_tensor - (1 - lambda_param) * max_similarities
            mmr_scores[selected_mask] = -float('inf')

            next_idx = torch.argmax(mmr_scores).item()
            next_similarity = max_similarities[next_idx].item()

            # Check threshold: if similarity is too high AND we've already selected min_remained_amount, stop.
            if len(selected_indices) >= min_remained_amount and next_similarity > similarity_threshold:
                # Next embedding exceeds similarity threshold—stop here.
                break

            # Otherwise, select embedding
            selected_indices.append(next_idx)
            selected_mask[next_idx] = True

            # Update max similarities
            max_similarities = torch.maximum(max_similarities, similarity_matrix[next_idx])

            pbar.update(1)

        pbar.close()

        return [indices[idx] for idx in selected_indices]

    def _distributed_mmr(self, embeddings, indices, scores, similarity_threshold,
                         lambda_param, min_remained_amount, batch_size, max_workers, main_device):
        """
        Distributed MMR filtering across multiple GPUs
        """
        import torch.multiprocessing as mp

        # Validate GPU count
        num_gpus = min(torch.cuda.device_count(), max_workers)
        if num_gpus <= 1:
            # Fall back to original method if only one GPU
            return self.mmr_diversity_filter(
                embeddings, indices, scores,
                similarity_threshold, lambda_param,
                min_remained_amount, batch_size, 1
            )

        # Split data across GPUs
        total_size = len(embeddings)
        chunk_size = total_size // num_gpus

        # Prepare queue for results
        manager = mp.Manager()
        result_queue = manager.Queue()

        # Prepare data for workers
        worker_args = []
        for i in range(num_gpus):
            chunk_start = i * chunk_size
            chunk_end = total_size if i == num_gpus - 1 else (i + 1) * chunk_size

            worker_args.append({
                'gpu_id': i,
                'embeddings': embeddings[chunk_start:chunk_end],
                'indices': indices[chunk_start:chunk_end],
                'scores': scores[chunk_start:chunk_end],
                'similarity_threshold': similarity_threshold,
                'lambda_param': lambda_param,
                'min_remained_amount': max(1, min_remained_amount // num_gpus),
                'batch_size': batch_size
            })

        # Worker function to process chunk on specific GPU
        def gpu_worker(worker_data, result_queue):
            try:
                import torch

                # Set GPU device
                torch.cuda.set_device(worker_data['gpu_id'])

                # Ensure PyTorch is using the correct device
                device = f'cuda:{worker_data["gpu_id"]}'

                # Create a temporary class instance
                from copy import deepcopy
                worker_instance = deepcopy(self)
                worker_instance.device = device

                # Run MMR on this chunk
                chunk_result = worker_instance.mmr_diversity_filter(
                    embeddings=worker_data['embeddings'],
                    indices=worker_data['indices'],
                    scores=worker_data['scores'],
                    similarity_threshold=worker_data['similarity_threshold'],
                    lambda_param=worker_data['lambda_param'],
                    min_remained_amount=worker_data['min_remained_amount'],
                    batch_size=worker_data['batch_size']
                )

                # Put results in queue
                result_queue.put(chunk_result)

            except Exception as e:
                import traceback
                logger.error(f"GPU {worker_data['gpu_id']} worker error: {e}")
                logger.error(traceback.format_exc())
                result_queue.put([])

        # Spawn processes for each GPU
        processes = []
        for args in worker_args:
            p = mp.Process(
                target=gpu_worker,
                args=(args, result_queue)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.extend(result_queue.get())

        # If no results, fall back to original method
        if not results:
            return indices

        # Perform final filtering if results exceed min_remained_amount
        if len(results) > min_remained_amount:
            final_result = self.mmr_diversity_filter(
                embeddings=[embeddings[indices.index(r)] for r in results],
                indices=results,
                scores=[scores[indices.index(r)] for r in results],
                similarity_threshold=similarity_threshold,
                lambda_param=lambda_param,
                min_remained_amount=min_remained_amount,
                batch_size=batch_size
            )
            return final_result

        return results

    import torch
    from tqdm import tqdm

    def mmr_incremental_multigpu(self, embeddings, indices, scores, similarity_threshold=0.95,
                                 lambda_param=0.5, min_remained_amount=None, gpu_devices=None, batch_size=20000,
                                 device='cuda'):

        if gpu_devices is None:
            gpu_devices = list(range(torch.cuda.device_count()))

        n_gpus = len(gpu_devices)
        device_ids = gpu_devices

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, dim=1)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        N = embeddings_tensor.size(0)
        selected_mask = torch.zeros(N, dtype=torch.bool)
        selected_indices = []

        if min_remained_amount is None:
            min_remained_amount = max(1, int(N * 0.1))

        current_idx = torch.argmax(scores_tensor).item()
        selected_indices.append(current_idx)
        selected_mask[current_idx] = True

        max_similarities = torch.zeros(N, dtype=torch.float32)

        pbar = tqdm(total=N)
        pbar.update(1)

        for _ in range(N - 1):
            candidate_indices = torch.nonzero(~selected_mask, as_tuple=False).flatten()
            candidate_scores = scores_tensor[candidate_indices]

            newly_selected_embedding = embeddings_tensor[selected_indices[-1]].unsqueeze(0)

            # Compute similarities incrementally in batches across multiple GPUs
            similarities = torch.zeros(len(candidate_indices))

            with torch.no_grad():
                for start in range(0, len(candidate_indices), batch_size * n_gpus):
                    end = min(start + batch_size * n_gpus, len(candidate_indices))
                    batch_candidates = candidate_indices[start:end]
                    split_batches = torch.chunk(batch_candidates, n_gpus)

                    gpu_results = []
                    for gpu_idx, gpu_batch in enumerate(split_batches):
                        gpu_device = device_ids[gpu_idx]
                        batch_embs = embeddings_tensor[gpu_batch].to(f'cuda:{gpu_device}', non_blocking=True)
                        selected_emb = newly_selected_embedding.to(f'cuda:{gpu_device}', non_blocking=True)

                        sim_batch = torch.mm(batch_embs, selected_emb.T).squeeze()
                        gpu_results.append(sim_batch.cpu())

                    similarities[start:end] = torch.cat(gpu_results)

            # Update max similarities
            max_similarities[candidate_indices] = torch.maximum(max_similarities[candidate_indices], similarities)

            # Compute MMR scores
            mmr_scores = lambda_param * scores_tensor - (1 - lambda_param) * max_similarities
            mmr_scores[selected_mask] = -float('inf')

            next_idx = torch.argmax(mmr_scores).item()
            next_similarity = max_similarities[next_idx].item()

            if len(selected_indices) >= min_remained_amount and next_similarity > similarity_threshold:
                break

            selected_indices.append(next_idx)
            selected_mask[next_idx] = True
            pbar.update(1)

        pbar.close()
        return [indices[idx] for idx in selected_indices]

