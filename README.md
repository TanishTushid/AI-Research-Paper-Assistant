# AI Research Paper Assistant

A local Flask web app for uploading research PDFs, asking questions with the
project's RAG pipeline, and viewing source citations.

## Run

1. Install the project requirements:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Start the Flask app:

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

3. Open <http://127.0.0.1:5000>.

Only PDFs uploaded through the web app are shown in the research library and
used to answer questions. Uploads are stored in `data/uploads/` and indexed
into the local FAISS vector store. Other PDFs already present in `data/` are
left untouched and excluded from the web app. The first question or upload may
take a little while as the local embedding and language models load. Uploads
are limited to PDF files up to 50 MB each.