# AI Research Paper Assistant

> A local-first research workspace for exploring PDF papers with semantic search, AI-generated answers, and page-level citations.

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/web-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Upload research PDFs, ask questions in plain language, and review the paper and page references behind each answer. Document processing, retrieval, and answer generation run locally.

## What you can do

- Upload one or more PDF papers to a personal research library.
- Search paper passages by meaning and ask questions grounded in your documents.
- Review source and page citations alongside answers.
- Remove a paper or clear the library whenever you like.
- Run the app with a local Qwen GGUF model and a FAISS vector index.

## How it works

1. **Upload:** PDFs are saved in `data/uploads/` and their text is extracted.
2. **Index:** Text is split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored in a local FAISS index.
3. **Ask:** The retriever finds relevant passages and sends them as context to the local Qwen model.
4. **Review:** The answer is shown with citations to the matching papers and pages.

The web library includes only papers uploaded through the app; PDFs elsewhere in `data/` are not automatically added.

## Quick start

### Requirements

- Python 3.14 or newer
- The local Qwen model file (download instructions below)
- Internet access on first run to download the embedding model

The project is configured for CPU inference. Initial setup and the first model load can take a few minutes.

### 1. Create an environment and install dependencies

In PowerShell:

```powershell
git clone https://github.com/TanishTushid/AI-Research-Paper-Assistant.git
cd AI-Research-Paper-Assistant

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run the app using `.\.venv\Scripts\python.exe` instead of `python`.

### 2. Download the local language model

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/llm
```

The model must be available at `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`. Model files are not included in the repository.

### 3. Start the app

```powershell
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Upload one or more PDFs (up to 50 MB each) to get started.

## Usage

- Drop PDF files onto the upload area or click it to browse.
- Enter a question and choose how many relevant passages to retrieve.
- Read the answer and check the listed paper/page citations.
- Use the library controls to remove individual papers or clear all uploaded papers.

## Project layout

```text
.
├── main.py                 # Flask web app
├── src/
│   ├── ingestion/          # PDF text extraction and chunking
│   ├── embeddings/         # Sentence-transformer embeddings
│   ├── retrieval/          # FAISS search and retrieval
│   ├── generation/         # Local GGUF model integration
│   ├── config.py           # Model and pipeline settings
│   └── pipeline.py         # RAG pipeline
├── templates/              # Web interface
├── static/                 # Front-end assets
├── data/uploads/            # Papers added through the app
├── models/llm/              # Local language model file
└── vectorstore/             # Generated FAISS index and metadata
```

## Local data and models

- Uploaded PDFs are stored in `data/uploads/`.
- The FAISS index and its metadata are stored in `vectorstore/`.
- The Qwen model is stored in `models/llm/`.
- The embedding model is downloaded by Sentence Transformers when first needed.

PDF processing, indexing, and question answering run on your machine. Model downloads require an internet connection. Generated model and vector-store files are excluded from version control.

## Configuration

Default model paths, retrieval settings, and CPU options are defined in [`src/config.py`](src/config.py). The web app accepts PDF files up to 50 MB each and supports retrieving between 1 and 10 passages per question.

## License

This project is released under the [MIT License](LICENSE).
