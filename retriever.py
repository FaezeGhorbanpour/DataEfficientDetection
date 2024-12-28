import os
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
        logging.info("Initializing the retriever module...")
        self.device = device
        self.index = self._initialize_index(embedding_dim, index_type)
        self.metadata = []
        logging.info(f"Retriever initialized with {index_type} index on {device}")

    def _initialize_index(self, embedding_dim, index_type):
        """
        Initialize FAISS index based on the given type.
        Args:
            embedding_dim (int): Dimension of embeddings.
            index_type (str): Type of FAISS index.
        Returns:
            faiss.Index: Initialized FAISS index.
        """
        logging.info(f"Creating index of type: {index_type}")
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
            logging.error(f"Unknown index_type: {index_type}")
            raise ValueError(f"Unknown index_type: {index_type}")

        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            logging.info("Using GPU for FAISS index.")
            return faiss.index_cpu_to_all_gpus(index)
        logging.info("Using CPU for FAISS index.")
        return index

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and their corresponding metadata to the index.
        Args:
            embeddings (np.ndarray): Embedding vectors to add (N x embedding_dim).
            metadata (list[dict]): Metadata associated with each embedding.
        """
        logging.info(f"Adding {len(embeddings)} embeddings to the index.")
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logging.info("Embeddings added successfully.")

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
        logging.info("Performing retrieval for a single query.")
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
        logging.info(f"Retrieved {len(results)} results.")
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
        logging.info("Starting retrieval for multiple queries.")

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
        results = [{"metadata": meta, "score": dist} for meta, dist in zip(metadata, flattened_distances)]

        # Deduplicate results by a unique key in metadata (assumes 'id' is the unique key)
        seen = set()
        deduplicated_results = []
        for result in results:
            unique_key = result["metadata"].get("dataset_name") + result["metadata"].get("id")  # Adjust to your unique metadata key
            if unique_key not in seen:
                seen.add(unique_key)
                deduplicated_results.append(result)

        logging.info(f"Total unique results after deduplication: {len(deduplicated_results)}")


        # Apply max_retrieved limit
        if max_retrieved is not None:
            # Sort deduplicated results by score (ascending order)
            deduplicated_results.sort(key=lambda x: x["score"])
            deduplicated_results = deduplicated_results[:max_retrieved]

        logging.info(f"Returning {len(deduplicated_results)} results after applying max_retrieved limit.")

        return deduplicated_results

    def save_meta_to_file(self, meta, path):
        """
        Saves a list of strings to a file, each string on a new line.

        Args:
            string_list (list): List of strings to save.
            file_path (str): Path to the file where the list will be saved.
        """

        try:
            file_path = os.path.join(path, "retrieved_data.json")
            with open(file_path, 'w') as json_file:
                json.dump(meta, json_file, indent=4)
            logging.info(f"List saved to {file_path}")
        except Exception as e:
            logging.info(f"An error occurred: {e}")

    def save_index(self, path):
        """
        Save the FAISS index and metadata to a file.
        Args:
            path (str): Directory to save the index and metadata.
        """
        logging.info(f"Saving index to {path}")
        if not os.path.exists(path):
            os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, 'embedding.index'))
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logging.info("Index saved successfully is this directory:")

    def load_index(self, path):
        """
        Load the FAISS index and metadata from a file.
        Args:
            path (str): Directory containing the index and metadata.
        """
        logging.info(f"Loading index from {path}")
        self.index = faiss.read_index(os.path.join(path, 'embedding.index'))
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        logging.info("Index loaded successfully.")
