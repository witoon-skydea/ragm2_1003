import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PERSIST_DIR.mkdir(exist_ok=True)

# Model configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
LLM_MODEL = "qwq:latest"  # Ollama model name

# Chunking configuration - ปรับให้เหมาะกับภาษาไทย
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ":", ";", " ", ""]

# Retrieval configuration
TOP_K = 3
HYBRID_SEARCH_WEIGHT = 0.7  # Weight for vector search in hybrid search (0.0-1.0)

# Vector DB configuration
HNSW_SPACE = "cosine"
HNSW_M = 16  # Maximum number of connections per node
HNSW_EF_CONSTRUCTION = 100  # ef during construction
HNSW_EF_SEARCH = 50  # ef during search