from pathlib import Path


# --------------------------------
# PROJECT DIRECTORIES
# --------------------------------

# Project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# --------------------------------
# LLM CONFIGURATION
# --------------------------------

LLM_MODEL_PATH = (
    LLM_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

LLM_CONTEXT_SIZE = 4096
LLM_THREADS = 4
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.2


# --------------------------------
# EMBEDDING CONFIGURATION
# --------------------------------

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cpu"


# --------------------------------
# CHUNKING CONFIGURATION
# --------------------------------

CHUNK_SIZE = 8
CHUNK_OVERLAP = 2


# --------------------------------
# RETRIEVAL CONFIGURATION
# --------------------------------

TOP_K = 5


# --------------------------------
# CREATE REQUIRED DIRECTORIES
# --------------------------------

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LLM_DIR,
    VECTORSTORE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)