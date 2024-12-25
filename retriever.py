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

    def retrieve_multiple_queries(self, query_embeddings, k=5, max_retrieved=None, filters=None):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings, then return the best `max_retrieved` results.
        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            max_retrieved (int): Maximum number of results to return overall across all queries.
            filters (dict): Optional metadata filters.
        Returns:
            list[dict]: Top results based on shortest distances, and the maximum number retrieved.
        """
        logging.info("Starting retrieval for multiple queries.")
        all_results = []

        for query_embedding in query_embeddings:
            distances, indices = self.index.search(query_embedding, k)

            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:  # No more valid indices
                    continue
                data_meta = self.metadata[idx]

                # Apply filters if provided
                if filters:
                    if not all(data_meta.get(key) == value for key, value in filters.items()):
                        continue

                # Collect results
                all_results.append({"metadata": data_meta, "score": dist})

        logging.info(f"Total results before sorting: {len(all_results)}")

        if max_retrieved and len(all_results) > max_retrieved:
            # Sort all results by score (ascending, since shorter distance is better)
            all_results = sorted(all_results, key=lambda x: x["score"])

            # Return only the top `max_retrieved` results
            if max_retrieved is not None:
                all_results = all_results[:max_retrieved]

        logging.info(f"Returning {len(all_results)} results after applying max_retrieved limit.")
        return all_results

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
