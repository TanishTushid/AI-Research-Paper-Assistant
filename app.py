from src.ingestion import load_all_documents, process_documents_to_chunks
from src.embeddings import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.generation import LLMModel

if __name__ == "__main__":
    # 1. Load documents 
    raw_docs = load_all_documents('data')
    
    # 2. Process and chunk documents using sentence-based strategy 
    chunked_docs = process_documents_to_chunks(raw_docs, max_chunk_chars=800, overlap_sentences=2)
    
    # Convert LangChain Document objects into the dictionary format expected by VectorStore
    chunks_dict_list = [
        {"text": doc.page_content, **doc.metadata} 
        for doc in chunked_docs
    ]
    
    # 3. Instantiate embedding model and generate embeddings 
    embed_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks_dict_list]
    embeddings = embed_model.generate_embeddings(texts)
    
    # 4. Initialize VectorStore, add embeddings, and save to disk
    vector_store = VectorStore(embedding_dimension=embed_model.embedding_dimension)
    vector_store.add_embeddings(chunks_dict_list, embeddings)
    vector_store.save("vectorstore")

    # 5. Initialize Retriever and test semantic search
    retriever = Retriever(embedding_model=embed_model, vector_store=vector_store)
    
    test_query = "What is the main methodology or architecture proposed?"
    context, results = retriever.get_context(test_query, top_k=3)
    
    print("\n--- Retrieved Context ---")
    print(context)

    # 6. LLM Testing & RAG Generation
    llm = LLMModel()
    
    # Example prompt combining the retrieved context and user query
    rag_prompt = f"""You are an AI Research Paper Assistant. Answer the question based ONLY on the following context. If the answer cannot be found, say "I cannot find the answer in the documents."

Context:
{context}

Question: {test_query}
Answer:"""

    response = llm.generate(prompt=rag_prompt)
    print("\n--- LLM RAG Response ---")
    print(response)

    print("\nPipeline completed successfully!")