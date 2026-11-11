from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class EmbeddingModel:
    """
    Generate text embeddings using a Sentence Transformer model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_name = model_name
        self.device = device

        # Load the embedding model
        self.model = SentenceTransformer(
            model_name,
            device=device
        )
        self.embedding_dimension = self.model.get_embedding_dimension()

        print(f"Embedding model loaded: {model_name} on {device}")
        print(f"Embedding dimension: {self.embedding_dimension}")

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Convert multiple text passages into embeddings.
        """
        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """
        Generate embeddings for chunks produced by chunking.py.
        """
        texts = [chunk["text"] for chunk in chunks]
        return self.generate_embeddings(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a user's question.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)