import os 
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    Docx2txtLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredPDFLoader, 
    JSONLoader, 
    UnstructuredMarkdownLoader, 
    UnstructuredPowerPointLoader, 
    UnstructuredHTMLLoader, 
    UnstructuredEPubLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from pathlib import Path
from typing import List, Any 


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported documents and convert to LangChain document format. 
    Supported formats include PDF, DOCX, TXT, JSON, Markdown, PowerPoint, HTML, EPUB, and Excel.
    """
    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")
    documents = []

    if not data_path.exists():
        print(f"Directory {data_path} does not exist.")
        return documents

    # 1. PDF files
    for pdf_file in data_path.glob("**/*.pdf"):
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF file {pdf_file}: {e}")

    # 2. Text files
    for text_file in data_path.glob("**/*.txt"):
        try:
            loader = TextLoader(str(text_file), encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading text file {text_file}: {e}")

    # 3. Word documents (.docx)
    for docx_file in data_path.glob("**/*.docx"):
        try:
            loader = Docx2txtLoader(str(docx_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading Word file {docx_file}: {e}")

    # 4. Markdown files (.md)
    for md_file in data_path.glob("**/*.md"):
        try:
            loader = UnstructuredMarkdownLoader(str(md_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading Markdown file {md_file}: {e}")

    # 5. JSON files (.json)
    # Note: JSONLoader requires a jq_schema depending on how the JSON is structured. 
    # Adjust jq_schema="." or point to a specific key if needed.
    for json_file in data_path.glob("**/*.json"):
        try:
            loader = JSONLoader(file_path=str(json_file), jq_schema=".", text_content=False)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading JSON file {json_file}: {e}")

    # 6. Excel spreadsheets (.xlsx, .xls)
    for excel_file in data_path.glob("**/*.xls*"):
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading Excel file {excel_file}: {e}")

    # 7. PowerPoint presentations (.pptx)
    for ppt_file in data_path.glob("**/*.ppt*"):
        try:
            loader = UnstructuredPowerPointLoader(str(ppt_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading PowerPoint file {ppt_file}: {e}")

    # 8. HTML files (.html, .htm)
    for html_file in data_path.glob("**/*.html"):
        try:
            loader = UnstructuredHTMLLoader(str(html_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading HTML file {html_file}: {e}")

    # 9. EPUB files (.epub)
    for epub_file in data_path.glob("**/*.epub"):
        try:
            loader = UnstructuredEPubLoader(str(epub_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading EPUB file {epub_file}: {e}")

    print(f"Successfully loaded {len(documents)} document chunks/pages in total.")
    return documents