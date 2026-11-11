import json
import logging
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.config import DATA_DIR, TOP_K, VECTORSTORE_DIR


LOGGER = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_REQUEST_BYTES = 250 * 1024 * 1024
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_pipeline = None
_pipeline_lock = threading.RLock()


def _is_upload_source(source: str | Path) -> bool:
    try:
        return Path(source).resolve().is_relative_to(UPLOAD_DIR.resolve())
    except (OSError, ValueError):
        return False


def _get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from src.pipeline import RAGPipeline

            _pipeline = RAGPipeline()
            obsolete_sources = {
                chunk.get("source") or chunk.get("file_path")
                for chunk in _pipeline.vector_store.metadata
                if (chunk.get("source") or chunk.get("file_path"))
                and (
                    not _is_upload_source(chunk.get("source") or chunk.get("file_path"))
                    or not Path(chunk.get("source") or chunk.get("file_path")).is_file()
                )
            }
            if obsolete_sources:
                _pipeline.vector_store.remove_sources(list(obsolete_sources))
                _pipeline.vector_store.save(str(VECTORSTORE_DIR))
        return _pipeline


def _normal_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _metadata_by_source() -> dict[str, dict]:
    metadata_path = VECTORSTORE_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Could not read vector-store metadata from %s", metadata_path)
        return {}

    documents = {}
    for chunk in chunks:
        source = chunk.get("source") or chunk.get("file_path")
        if source:
            documents.setdefault(_normal_path(source), chunk)
    return documents


def _document_list() -> list[dict]:
    indexed = _metadata_by_source()
    documents = []
    for path in sorted(UPLOAD_DIR.rglob("*"), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in ALLOWED_EXTENSIONS:
            continue
        details = indexed.get(_normal_path(path), {})
        relative_path = path.relative_to(UPLOAD_DIR).as_posix()
        documents.append({
            "id": relative_path,
            "name": path.name,
            "size": path.stat().st_size,
            "pages": details.get("total_pages"),
            "indexed": bool(details),
        })
    return documents


def _resolve_document(document_id: str) -> Path | None:
    for document in _document_list():
        if document["id"] == document_id:
            candidate = (UPLOAD_DIR / Path(document_id)).resolve()
            if candidate.is_relative_to(UPLOAD_DIR.resolve()):
                return candidate
    return None


def _json_error(message: str, status: int):
    return jsonify({"error": message}), status


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

    @app.get("/")
    def home():
        return render_template("index.html", top_k=TOP_K)

    @app.get("/api/documents")
    def documents():
        return jsonify({"documents": _document_list()})

    @app.post("/api/upload")
    def upload():
        uploads = request.files.getlist("files")
        if not uploads or all(not item.filename for item in uploads):
            return _json_error("Choose one or more PDF files to upload.", 400)

        results = []
        with _pipeline_lock:
            for uploaded in uploads:
                original_name = uploaded.filename or ""
                safe_name = secure_filename(original_name)
                if not safe_name or Path(safe_name).suffix.casefold() not in ALLOWED_EXTENSIONS:
                    results.append({"name": original_name, "error": "Only PDF files are supported."})
                    continue
                if not uploaded.stream.seekable():
                    results.append({"name": original_name, "error": "The uploaded file could not be read."})
                    continue
                uploaded.stream.seek(0, os.SEEK_END)
                file_size = uploaded.stream.tell()
                uploaded.stream.seek(0)
                if file_size == 0:
                    results.append({"name": original_name, "error": "The PDF is empty."})
                    continue
                if file_size > MAX_UPLOAD_BYTES:
                    results.append({"name": original_name, "error": "The file exceeds the 50 MB limit."})
                    continue
                if uploaded.stream.read(5) != b"%PDF-":
                    results.append({"name": original_name, "error": "The file does not appear to be a valid PDF."})
                    continue
                uploaded.stream.seek(0)

                target = UPLOAD_DIR / safe_name
                if target.exists():
                    results.append({"name": original_name, "error": "A document with this filename already exists."})
                    continue
                uploaded.save(target)

                try:
                    from langchain_community.document_loaders import PyMuPDFLoader

                    pages = PyMuPDFLoader(str(target)).load()
                    if not pages:
                        raise ValueError("No readable pages were found in the PDF.")
                    _get_pipeline().ingest_documents(pages)
                    results.append({"name": safe_name, "success": True})
                except Exception as exc:
                    target.unlink(missing_ok=True)
                    LOGGER.exception("Failed to index uploaded PDF %s", safe_name)
                    results.append({"name": original_name, "error": f"Could not index this PDF: {exc}"})

        failed = [result for result in results if "error" in result]
        status = 400 if len(failed) == len(results) else 200
        return jsonify({
            "results": results,
            "documents": _document_list(),
        }), status

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return _json_error("Enter a question about your research papers.", 400)
        try:
            top_k = int(payload.get("top_k", TOP_K))
        except (TypeError, ValueError):
            return _json_error("The result count must be a whole number.", 400)
        if top_k < 1 or top_k > 10:
            return _json_error("Choose between 1 and 10 results.", 400)

        try:
            with _pipeline_lock:
                response = _get_pipeline().ask(question.strip(), top_k=top_k)
        except Exception as exc:
            LOGGER.exception("Question answering failed")
            return _json_error(f"Could not answer the question: {exc}", 500)

        citations = []
        for citation in response.get("citations", []):
            source = Path(str(citation.get("source", "Unknown source"))).name
            try:
                page = int(citation.get("page"))
                page = page + 1 if page >= 0 else page
            except (TypeError, ValueError):
                page = citation.get("page", "Unknown page")
            citations.append({"source": source, "page": page})
        return jsonify({
            "answer": response.get("answer", ""),
            "citations": citations,
        })

    @app.delete("/api/documents/<path:document_id>")
    def delete_document(document_id: str):
        target = _resolve_document(document_id)
        if target is None:
            return _json_error("That document was not found in the research library.", 404)

        try:
            with _pipeline_lock:
                pipeline = _get_pipeline()
                pipeline.vector_store.remove_sources([str(target)])
                pipeline.vector_store.save(str(VECTORSTORE_DIR))
                target.unlink()
        except Exception as exc:
            LOGGER.exception("Could not remove document %s", document_id)
            return _json_error(f"Could not remove the document: {exc}", 500)
        return jsonify({"documents": _document_list()})

    @app.delete("/api/documents")
    def clear_documents():
        targets = [path for path in UPLOAD_DIR.rglob("*")
                   if path.is_file() and path.suffix.casefold() in ALLOWED_EXTENSIONS]
        try:
            with _pipeline_lock:
                pipeline = _get_pipeline()
                pipeline.vector_store.remove_sources([str(path) for path in targets])
                pipeline.vector_store.save(str(VECTORSTORE_DIR))
                for path in targets:
                    path.unlink()
        except Exception as exc:
            LOGGER.exception("Could not clear the research library")
            return _json_error(f"Could not clear the research library: {exc}", 500)
        return jsonify({"documents": _document_list()})

    @app.errorhandler(413)
    def upload_too_large(_error):
        LOGGER.warning("Rejected oversized upload request: %s", _error)
        return _json_error("The upload request exceeds the 250 MB total limit.", 413)

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
