import os
import random

import faiss
import numpy as np
from datasets import Dataset
import json
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedding_dim=768, index_type="FlatL2", device="cuda"):
        """
        Initialize the Retriever with an FAISS index.
        Args:
            embedding_dim (int): Dimension of the embeddings.
            index_type (str): FAISS index type (e.g., "FlatL2", "IVF", "HNSW").
            device (str): Device to use ("cuda" or "cpu").
        """
        logger.info("Initializing the retriever module...")
        self.device = device
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
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(embedding_dim, 128)
            hnswpq_index = faiss.downcast_index(index)
            hnswpq_index.hnsw.efConstruction = 200
            hnswpq_index.hnsw.efSearch = 128
        elif index_type == "IVF":
            index = faiss.IndexIVFPQ(faiss.IndexFlatL2(embedding_dim), embedding_dim, 100, 32, 32)
        else:
            logger.error(f"Unknown index_type: {index_type}")
            raise ValueError(f"Unknown index_type: {index_type}")

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
        self.index.add(embeddings)
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

    def retrieve_multiple_queries(self, query_embeddings, k=5, max_retrieved=None, exclude_datasets=None, exclude_languages=None):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings, with optional filtering and deduplication.
        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            max_retrieved (int): Maximum number of results to return overall across all queries.
            filters (dict): Optional metadata filters.
        Returns:
            list[dict]: Top results based on shortest distances, with optional filtering and deduplication.
        """
        logger.info("Starting retrieval for multiple queries.")

        # Perform the search for all queries at once
        distances, indices = self.index.search(query_embeddings, k)

        # Flatten distances and indices for efficient processing
        num_queries = query_embeddings.shape[0]
        flattened_distances = distances.flatten()
        flattened_indices = indices.flatten()

        # Filter out invalid indices (-1)
        valid_mask = flattened_indices != -1
        flattened_distances = flattened_distances[valid_mask]
        flattened_indices = flattened_indices[valid_mask]

        # Fetch metadata for all valid indices
        metadata = [self.metadata[idx] for idx in flattened_indices]

        # Apply filters if provided
        if exclude_languages:
            filter_mask = [
                all(meta.get('language') == value for value in exclude_languages) for meta in metadata
            ]
            flattened_distances = flattened_distances[filter_mask]
            metadata = [meta for meta, keep in zip(metadata, filter_mask) if keep]
        # Apply filters if provided
        if exclude_datasets:
            filter_mask = [
                all(meta.get('dataset_name') != value for value in exclude_datasets) for meta in metadata
            ]
            flattened_distances = flattened_distances[filter_mask]
            metadata = [meta for meta, keep in zip(metadata, filter_mask) if keep]

        # Combine distances and metadata into a single structure
        results = [{"metadata": meta, "score": float(dist)} for meta, dist in zip(metadata, flattened_distances)]

        # Deduplicate results by a unique key in metadata (assumes 'id' is the unique key)
        seen = set()
        deduplicated_results = []
        for result in results:
            unique_key = result["metadata"].get("dataset_name") + result["metadata"].get("id")  # Adjust to your unique metadata key
            if unique_key not in seen:
                seen.add(unique_key)
                deduplicated_results.append(result)

        logger.info(f"Total unique results after deduplication: {len(deduplicated_results)}")


        # Apply max_retrieved limit
        if max_retrieved is not None:
            # Sort deduplicated results by score (ascending order)
            deduplicated_results.sort(key=lambda x: x["score"])
            deduplicated_results = deduplicated_results[:max_retrieved]

        logger.info(f"Returning {len(deduplicated_results)} results after applying max_retrieved limit.")

        return deduplicated_results


    def retrieve_random_metadata(self, max_retrieved=None, exclude_datasets=None, exclude_languages=None):
        """
        Retrieve a random selection of metadata, with optional filtering.

        Args:
            num_results (int): Number of random metadata entries to retrieve.
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
        random_metadata = dict()
        random_metadata['metadata'] = random.sample(filtered_metadata, num_results)

        logger.info(f"Returning {len(random_metadata)} random metadata entries.")

        return random_metadata

    def save_meta_to_file(self, meta, path):
        """
        Saves a list of strings to a file, each string on a new line.

        Args:
            string_list (list): List of strings to save.
            file_path (str): Path to the file where the list will be saved.
        """

        try:
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, "retrieved_data.json")
            with open(file_path, 'w') as json_file:
                json.dump(meta, json_file, indent=4, ensure_ascii=False)
            logger.info(f"List saved to {file_path}")
        except Exception as e:
            logger.info(f"An error occurred: {e}")

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
        logger.info("Index saved successfully is this directory:")

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
