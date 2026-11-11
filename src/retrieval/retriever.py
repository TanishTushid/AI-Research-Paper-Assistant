
class Retriever:
    """
    Retrieve relevant document chunks using semantic search.
    """

    def __init__(self, embedding_model, vector_store):

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Convert the user's question into an embedding
        and retrieve the most relevant chunks from FAISS.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Step 1: Convert question into embedding
        query_embedding = self.embedding_model.embed_query(question)

        # Step 2: Search FAISS vector database
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Step 3: Return retrieved chunks
        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Convert retrieved chunks into readable context
        for the LLM, preserving source information.
        """

        if not results:
            return "No relevant information was found in the documents."

        context_parts = []

        for index, result in enumerate(results, start=1):

            text = result.get("text", "")

            metadata = result.get("metadata", {})

            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown page")
            chunk_id = metadata.get("chunk_id", "Unknown chunk")

            context = f"""
[Context {index}]
Source: {source}
Page: {page}
Chunk ID: {chunk_id}

Content:
{text}
"""

            context_parts.append(context.strip())

        return "\n\n".join(context_parts)

    def get_context(
        self,
        question: str,
        top_k: int = 5
    ) -> tuple[str, list[dict]]:
        """
        Retrieve relevant chunks and return formatted context
        along with the original results and metadata.
        """

        results = self.retrieve(question, top_k)

        context = self.format_context(results)

        return context, results
