from src.pipeline import RAGPipeline
from src.config import DATA_DIR
from src.ingestion import load_all_documents

if __name__ == "__main__":
    print("Starting RAG test pipeline...\n")

    # 1. Initialize the RAG Pipeline (loads embedding model, vector store, and LLM)
    rag = RAGPipeline()

    # 2. Load raw documents from the data directory
    print(f"\nLoading documents from: {DATA_DIR}")
    documents = load_all_documents(str(DATA_DIR))

    if documents:
        # 3. Ingest documents (chunks text, generates embeddings, and saves to FAISS)
        num_chunks = rag.ingest_documents(documents)
        print(f"Successfully ingested {num_chunks} chunks into the vector store.\n")
    else:
        print("No documents found in the data directory. Using existing vector store (if available).\n")

    # 4. Test asking a question
    test_query = "What is the main methodology or architecture proposed?"
    print(f"Query: {test_query}\n")

    response = rag.ask(test_query)

    print("\n--- Final Answer ---")
    print(response["answer"])

    print("\n--- Citations ---")
    if response["citations"]:
        for citation in response["citations"]:
            print(f"- Source: {citation['source']}, Page: {citation['page']}")
    else:
        print("No citations found.")

    print("\nTest pipeline completed successfully!")