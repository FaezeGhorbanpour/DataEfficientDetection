import os.path

import faiss
import numpy as np
from datasets import Dataset
import json


class Retriever:
    def __init__(self, embedding_dim=768, index_type="FlatL2", device="cuda"):
        """
        Initialize the Retriever with an FAISS index.
        Args:
            embedding_dim (int): Dimension of the embeddings.
            index_type (str): FAISS index type (e.g., "FlatL2", "IVF", "HNSW").
            device (str): Device to use ("cuda" or "cpu").
        """
        self.device = device
        self.index = self._initialize_index(embedding_dim, index_type)
        self.metadata = []  # To store metadata for filtering during retrieval

    def _initialize_index(self, embedding_dim, index_type):
        """
        Initialize FAISS index based on the given type.
        Args:
            embedding_dim (int): Dimension of embeddings.
            index_type (str): Type of FAISS index.
        Returns:
            faiss.Index: Initialized FAISS index.
        """
        if index_type == "FlatL2":
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(embedding_dim, 128)  # 32 is the number of neighbors
        elif index_type == "IVF":
            index = faiss.IndexIVFPQ(faiss.IndexFlatL2(embedding_dim), embedding_dim, 100, 32, 32)#IndexIVFPQ
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            print('Using CUDA device')
            return faiss.index_cpu_to_all_gpus(index)
        return index

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and their corresponding metadata to the index.
        Args:
            embeddings (np.ndarray): Embedding vectors to add (N x embedding_dim).
            metadata (list[dict]): Metadata associated with each embedding.
        """
        self.index.add(embeddings)
        self.metadata.extend(metadata)

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
        # Ensure the query embedding is normalized for cosine similarity if needed
        distances, indices = self.index.search(query_embedding, k)

        # Filter by metadata if filters are provided
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # No more valid indices
                continue
            data_meta = self.metadata[idx]
            if filters:
                if not all(data_meta.get(key) == value for key, value in filters.items()):
                    continue
            results.append({"metadata": data_meta, "score": dist})
        return results[:k]

    def retrieve_multiple_queries(self, query_embeddings, k=5, filters=None):
        """
        Retrieve top-k nearest neighbors for multiple query embeddings, ensuring uniqueness in results.
        Args:
            query_embeddings (np.ndarray): Query embeddings (N x embedding_dim).
            k (int): Number of nearest neighbors to retrieve for each query.
            filters (dict): Optional metadata filters.
        Returns:
            list[dict]: Combined list of metadata and scores for all queries.
        """
        all_results = []
        seen_metadata = set()  # To ensure unique metadata entries

        for query_embedding in query_embeddings:
            distances, indices = self.index.search(query_embedding, k)

            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:  # No more valid indices
                    continue
                data_meta = self.metadata[idx]

                # Filter by metadata if filters are provided
                if filters:
                    if not all(data_meta.get(key) == value for key, value in filters.items()):
                        continue

                # Ensure uniqueness in the results
                if data_meta not in seen_metadata:
                    seen_metadata.add(data_meta)
                    all_results.append({"metadata": data_meta, "score": dist})

        return all_results[:k]  # Limit results to the maximum k

    def save_index(self, path):
        """Save the FAISS index to a file."""
        faiss.write_index(self.index, os.path.join(path, 'embedding.index'))
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load_index(self, path):
        """Load the FAISS index from a file."""
        self.index = faiss.read_index(os.path.join(path, 'embedding.index'))
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

