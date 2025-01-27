import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class MultilingualRetriever:
    def __init__(self, index, metadata):
        """
        Args:
            index: FAISS or similar index for fast retrieval.
            metadata: List of metadata dictionaries corresponding to indexed embeddings.
        """
        self.index = index
        self.metadata = metadata

    def retrieve_multiple_queries(self, query_embeddings, k=500, max_retrieved=None, exclude_datasets=None, exclude_languages=None, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings with multi-criteria filtering and scoring.
        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            max_retrieved (int): Maximum number of results to return overall across all queries.
            exclude_datasets (list): Datasets to exclude.
            exclude_languages (list): Languages to exclude.
            alpha (float): Weight for similarity.
            beta (float): Weight for diversity.
            gamma (float): Weight for informativeness.
        Returns:
            list[dict]: Top results based on multi-criteria scoring.
        """
        logger.info("Starting multi-criteria retrieval for multiple queries.")

        # Perform the search for all queries
        distances, indices = self.index.search(query_embeddings, k)

        # Flatten distances and indices for processing
        num_queries = query_embeddings.shape[0]
        flattened_distances = distances.flatten()
        flattened_indices = indices.flatten()

        # Filter out invalid indices (-1)
        valid_mask = flattened_indices != -1
        flattened_distances = flattened_distances[valid_mask]
        flattened_indices = flattened_indices[valid_mask]

        # Fetch metadata for valid indices
        metadata = [self.metadata[idx] for idx in flattened_indices]

        # Apply filters
        if exclude_languages:
            metadata, flattened_distances = self._filter_metadata(metadata, flattened_distances, 'language', exclude_languages)
        if exclude_datasets:
            metadata, flattened_distances = self._filter_metadata(metadata, flattened_distances, 'dataset_name', exclude_datasets)

        # Create initial results structure
        results = [
            {
                "metadata": meta,
                "score": float(dist),
                "embedding": self.index.reconstruct(idx),
            }
            for meta, dist, idx in zip(metadata, flattened_distances, flattened_indices)
        ]

        # Add multi-criteria scoring
        results = self._apply_multi_criteria_scoring(results, query_embeddings, alpha, beta, gamma)

        # Deduplicate results
        results = self._deduplicate_results(results)

        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Apply max_retrieved limit
        if max_retrieved is not None:
            results = results[:max_retrieved]

        logger.info(f"Returning {len(results)} results after applying multi-criteria scoring and deduplication.")
        return results

    def _filter_metadata(self, metadata, distances, key, exclude_values):
        """Filter metadata based on exclude values."""
        mask = [meta.get(key) not in exclude_values for meta in metadata]
        filtered_metadata = [meta for meta, keep in zip(metadata, mask) if keep]
        filtered_distances = distances[mask]
        return filtered_metadata, filtered_distances

    def _apply_multi_criteria_scoring(self, results, query_embeddings, alpha, beta, gamma):
        """Apply multi-criteria scoring to results."""
        query_mean = np.mean(query_embeddings, axis=0)

        # Compute similarity, diversity, and informativeness
        for result in results:
            embedding = result["embedding"]
            similarity = cosine_similarity([query_mean], [embedding])[0][0]

            diversity = self._compute_diversity(result, results)
            informativeness = self._compute_informativeness(result)

            # Weighted scoring
            result["similarity"] = similarity
            result["diversity"] = diversity
            result["informativeness"] = informativeness
            result["final_score"] = alpha * similarity + beta * diversity + gamma * informativeness

        return results

    def _compute_diversity(self, result, results):
        """Compute diversity score using clustering."""
        embeddings = np.array([res["embedding"] for res in results])
        kmeans = KMeans(n_clusters=min(10, len(embeddings)), random_state=42).fit(embeddings)
        cluster_label = kmeans.predict([result["embedding"]])[0]
        cluster_size = sum(1 for lbl in kmeans.labels_ if lbl == cluster_label)
        return 1 / cluster_size  # Smaller clusters are more diverse

    def _compute_informativeness(self, result):
        """Compute informativeness using prediction entropy."""
        prediction_probs = result["metadata"].get("prediction_probs", [0.5, 0.5])  # Example placeholder
        return entropy(prediction_probs)

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


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logger = logging.getLogger(__name__)


class MultilingualRetriever:
    def __init__(self, index, metadata, model_name="cardiffnlp/twitter-roberta-base-hate"):
        """
        Initialize the retriever with the index, metadata, and optional prediction model.

        Args:
            index: FAISS or similar index for nearest neighbor search.
            metadata (list[dict]): Metadata associated with the indexed embeddings.
            model_name (str): Pretrained hate speech detection model name.
        """
        self.index = index
        self.metadata = metadata

        # Load the hate speech detection model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def add_prediction_probs(self, texts):
        """
        Add prediction probabilities to metadata using a pretrained model.

        Args:
            texts (list[str]): List of texts corresponding to metadata entries.

        Returns:
            list[dict]: Updated metadata with prediction probabilities.
        """
        updated_metadata = []
        with torch.no_grad():
            for meta, text in zip(self.metadata, texts):
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

                meta_copy = meta.copy()
                meta_copy["prediction_probs"] = probs.tolist()
                updated_metadata.append(meta_copy)

        self.metadata = updated_metadata

    def _compute_informativeness(self, metadata):
        """
        Compute informativeness based on prediction probabilities.

        Args:
            metadata (dict): Metadata containing prediction probabilities.

        Returns:
            float: Informativeness score.
        """
        probs = np.array(metadata.get("prediction_probs", []))
        return 1 - np.max(probs)  # Lower max probability indicates more uncertainty/informativeness

    def _compute_diversity(self, selected_embeddings, candidate_embedding):
        """
        Compute diversity of a candidate embedding compared to selected embeddings.

        Args:
            selected_embeddings (list[np.ndarray]): List of already selected embeddings.
            candidate_embedding (np.ndarray): Embedding of the candidate.

        Returns:
            float: Diversity score.
        """
        if not selected_embeddings:
            return 1.0

        similarities = cosine_similarity([candidate_embedding], selected_embeddings)[0]
        return 1 - np.mean(similarities)  # Higher mean similarity reduces diversity

    def _apply_multi_criteria_scoring(self, results, selected_embeddings):
        """
        Apply multi-criteria scoring to rank results by informativeness and diversity.

        Args:
            results (list[dict]): List of results with metadata and embeddings.
            selected_embeddings (list[np.ndarray]): List of already selected embeddings.

        Returns:
            list[dict]: Scored and sorted results.
        """
        for result in results:
            informativeness = self._compute_informativeness(result["metadata"])
            diversity = self._compute_diversity(selected_embeddings, result["metadata"]["embedding"])
            result["score"] = informativeness + diversity

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _deduplicate_results(self, results):
        """
        Deduplicate results based on text content.

        Args:
            results (list[dict]): List of results with metadata.

        Returns:
            list[dict]: Deduplicated results.
        """
        seen_texts = set()
        deduplicated_results = []

        for result in results:
            text = result["metadata"].get("text")
            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated_results.append(result)

        return deduplicated_results

    def retrieve_multiple_queries(self, query_embeddings, k=500, max_retrieved=None, exclude_languages=None):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings, enforcing diversity and informativeness.

        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            max_retrieved (int): Maximum number of results to return overall across all queries.
            exclude_languages (list[str]): Languages to exclude from results.

        Returns:
            list[dict]: Retrieved results with diversity and optional filtering applied.
        """
        logger.info("Starting retrieval for multiple queries.")

        distances, indices = self.index.search(query_embeddings, k)

        all_results = []
        for query_idx in range(len(query_embeddings)):
            query_distances = distances[query_idx]
            query_indices = indices[query_idx]

            valid_mask = query_indices != -1
            query_distances = query_distances[valid_mask]
            query_indices = query_indices[valid_mask]

            query_metadata = [self.metadata[idx] for idx in query_indices]

            if exclude_languages:
                filter_mask = [
                    meta.get('language') not in exclude_languages for meta in query_metadata
                ]
                query_metadata = [meta for meta, keep in zip(query_metadata, filter_mask) if keep]
                query_distances = query_distances[filter_mask]

            results = [
                {"metadata": meta, "score": float(dist), "embedding": self.index.reconstruct(idx)}
                for meta, dist, idx in zip(query_metadata, query_distances, query_indices)
            ]

            all_results.extend(results)

        all_results = self._deduplicate_results(all_results)
        selected_embeddings = []
        final_results = []

        while all_results and (max_retrieved is None or len(final_results) < max_retrieved):
            all_results = self._apply_multi_criteria_scoring(all_results, selected_embeddings)
            best_result = all_results.pop(0)

            final_results.append(best_result)
            selected_embeddings.append(best_result["embedding"])

        logger.info(f"Returning {len(final_results)} diverse and informative results.")
        return final_results
