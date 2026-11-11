from pathlib import Path

from src.config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_DEVICE,
    LLM_MODEL_PATH,
    LLM_CONTEXT_SIZE,
    LLM_THREADS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)

from src.ingestion.chunking import process_documents_to_chunks
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation import LLMModel


class RAGPipeline:
    """
    Main pipeline connecting document ingestion,
    embeddings, FAISS retrieval and LLM generation.
    """

    def __init__(self):
        print("Initializing RAG pipeline...")

        # Load embedding model
        self.embedding_model = EmbeddingModel(
            model_name=EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE
        )

        # Initialize vector database
        self.vector_store = VectorStore(
            embedding_dimension=EMBEDDING_DIMENSION
        )

        # Load existing vector database if available
        index_path = VECTORSTORE_DIR / "faiss.index"
        metadata_path = VECTORSTORE_DIR / "metadata.json"

        if index_path.exists() and metadata_path.exists():
            self.vector_store.load(str(VECTORSTORE_DIR))
            print("Existing vector database loaded.")

        elif index_path.exists() or metadata_path.exists():
            raise FileNotFoundError(
                "Incomplete vector database: faiss.index "
                "or metadata.json is missing."
            )

        else:
            print("No existing vector database found. Starting fresh.")

        # Initialize retriever
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        # Load local LLM
        self.llm_model = LLMModel(
            model_path=LLM_MODEL_PATH,
            context_size=LLM_CONTEXT_SIZE,
            n_threads=LLM_THREADS
        )

        print("RAG pipeline initialized successfully!")

    def ingest_documents(self, documents: list) -> int:
        """
        Process documents and store their embeddings in FAISS.
        """
        if not documents:
            raise ValueError("No documents provided for ingestion.")

        print("Creating text chunks...")

        # Step 1: Process documents into sentence chunks
        chunked_docs = process_documents_to_chunks(
            documents,
            max_chunk_chars=CHUNK_SIZE,
            overlap_sentences=CHUNK_OVERLAP
        )

        if not chunked_docs:
            raise ValueError("No valid chunks were generated.")

        # Convert LangChain Document objects into dictionary format for VectorStore
        chunks_dict_list = [
            {"text": doc.page_content, **doc.metadata} 
            for doc in chunked_docs
        ]

        print(f"Created {len(chunks_dict_list)} chunks.")

        # Step 2: Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.embed_chunks(chunks_dict_list)

        # Step 3: Store embeddings in FAISS
        print("Adding embeddings to FAISS...")
        self.vector_store.add_embeddings(
            chunks=chunks_dict_list,
            embeddings=embeddings
        )

        # Step 4: Save vector database
        self.vector_store.save(str(VECTORSTORE_DIR))

        print("Document ingestion completed!")
        return len(chunks_dict_list)

    def ask(
        self,
        question: str,
        top_k: int = TOP_K
    ) -> dict:
        """
        Retrieve relevant context and generate an answer.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        print("Retrieving relevant documents...")

        # Step 1: Retrieve relevant chunks
        context, results = self.retriever.get_context(
            question=question,
            top_k=top_k
        )

        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "citations": [],
                "retrieved_chunks": []
            }

        # Step 2: Generate answer using LLM
        print("Generating answer...")
        answer = self.llm_model.answer_question(
            question=question,
            context=context
        )

        # Step 3: Extract citations from original metadata
        citations = []
        seen_citations = set()

        for result in results:
            metadata = result.get("metadata", result)
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown page")

            citation_key = (source, page)

            if citation_key not in seen_citations:
                citations.append({
                    "source": source,
                    "page": page
                })
                seen_citations.add(citation_key)

        # Step 4: Return answer and references
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": results
        }