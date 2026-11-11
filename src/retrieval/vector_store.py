

import os
import json
import faiss
import numpy as np


class VectorStore:
    """
    Store and retrieve text embeddings using FAISS.
    """

    def __init__(self, embedding_dimension: int = 384):

        self.embedding_dimension = embedding_dimension

        # Inner product index for normalized vectors
        self.index = faiss.IndexFlatIP(embedding_dimension)

        # Stores original text and metadata
        # Position in this list matches vector position in FAISS
        self.metadata = []

    def add_embeddings(
        self,
        chunks: list[dict],
        embeddings: np.ndarray
    ):
        """
        Add embeddings and corresponding chunks to FAISS.
        """

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be equal"
            )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError("Embedding dimension mismatch")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain invalid values")

        embeddings = np.ascontiguousarray(embeddings)

        # Add vectors to FAISS
        self.index.add(embeddings)

        # Store corresponding text and metadata
        self.metadata.extend(chunks)

        print(f"Added {len(chunks)} embeddings to FAISS.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[dict]:
        """
        Find the most similar chunks for a query embedding.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.embedding_dimension):
            raise ValueError("Query embedding dimension mismatch")

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding contains invalid values")

        # Search cannot return more results than stored vectors
        top_k = min(top_k, self.index.ntotal)

        query_embedding = np.ascontiguousarray(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            result = self.metadata[idx].copy()

            result["score"] = float(score)

            results.append(result)

        return results

    def remove_sources(self, sources: list[str]) -> int:
        """Remove indexed chunks whose source path is in ``sources``."""
        normalized_sources = {
            os.path.normcase(os.path.abspath(source))
            for source in sources
        }
        keep_indices = []
        kept_metadata = []

        for index, chunk in enumerate(self.metadata):
            source = chunk.get("source") or chunk.get("file_path")
            normalized_source = (
                os.path.normcase(os.path.abspath(source))
                if source else ""
            )
            if normalized_source not in normalized_sources:
                keep_indices.append(index)
                kept_metadata.append(chunk)

        removed_count = len(self.metadata) - len(kept_metadata)
        if not removed_count:
            return 0

        kept_embeddings = [
            self.index.reconstruct(index)
            for index in keep_indices
        ]
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        if kept_embeddings:
            self.index.add(np.ascontiguousarray(kept_embeddings, dtype=np.float32))
        self.metadata = kept_metadata
        return removed_count

    def save(self, directory: str = "vectorstore"):
        """
        Save FAISS index and metadata to disk.
        """

        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, "faiss.index")
        metadata_path = os.path.join(directory, "metadata.json")

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=4
            )

        print(f"Vector store saved at: {directory}")

    def load(self, directory: str = "vectorstore"):
        """
        Load FAISS index and metadata from disk.
        """

        index_path = os.path.join(directory, "faiss.index")
        metadata_path = os.path.join(directory, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)

        loaded_index = faiss.read_index(index_path)

        if loaded_index.d != self.embedding_dimension:
            raise ValueError("Saved index dimension mismatch")

        with open(metadata_path, "r", encoding="utf-8") as file:
            loaded_metadata = json.load(file)

        if loaded_index.ntotal != len(loaded_metadata):
            raise ValueError("Index and metadata count mismatch")

        self.index = loaded_index
        self.metadata = loaded_metadata

        print(f"Vector store loaded from: {directory}")
