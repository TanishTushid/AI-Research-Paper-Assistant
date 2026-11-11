import re
from typing import List
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean raw extracted text by removing excessive whitespace, 
    unwanted line breaks, and formatting artifacts.
    """
    if not text:
        return ""
    
    # Replace multiple spaces, newlines, and tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Split clean text into individual sentences using a robust regex pattern 
    targeting punctuation marks followed by whitespace.
    """
    if not text:
        return []
    
    # Regex split on sentence boundaries (. ! ?) followed by whitespace
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    
    # Filter out empty strings or whitespace-only items
    return [s.strip() for s in sentences if s.strip()]


def create_sentence_chunks(
    sentences: List[str], 
    max_chunk_chars: int = 800, 
    overlap_sentences: int = 1
) -> List[str]:
    """
    Group sentences into chunks based on a maximum character size limit,
    incorporating a sentence-level overlap to preserve context across boundaries.
    """
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds the max size and we already have content, store chunk
        if current_length + sentence_len > max_chunk_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep overlap sentences from the end of the previous chunk
            overlap_pool = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = list(overlap_pool)
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_documents_to_chunks(
    documents: List[Document], 
    max_chunk_chars: int = 800, 
    overlap_sentences: int = 1
) -> List[Document]:
    """
    Main pipeline function:
    Extracted documents -> clean_text() -> split_sentences() -> 
    create_sentence_chunks() -> Preserve metadata -> Final chunks
    """
    print(f"Starting sentence-based chunking for {len(documents)} source documents...")
    final_chunks = []

    for doc in documents:
        raw_text = doc.page_content
        metadata = doc.metadata.copy()

        # Clean text
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        # Split into sentences
        sentences = split_sentences(cleaned)

        # Create sentence-based chunks with overlap
        text_chunks = create_sentence_chunks(
            sentences, 
            max_chunk_chars=max_chunk_chars, 
            overlap_sentences=overlap_sentences
        )

        # Preserve metadata and build final LangChain Document chunks
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            
            final_chunk_doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            final_chunks.append(final_chunk_doc)

    print(f"Successfully generated {len(final_chunks)} sentence-based chunks ready for embedding.")
    return final_chunks