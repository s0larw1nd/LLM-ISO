import os

__parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULT_MODEL = os.path.join(__parent_dir, "models", "Qwen2.5-7B-Instruct-merged")
DEFAULT_EMBEDDINGS = os.path.join(__parent_dir, "models", "Qwen3-Embedding-0.6B")
DEFAULT_DB_DIR = os.path.join(__parent_dir, "db", "chroma_db")
DEFAULT_DOC_DIR = os.path.join(__parent_dir, "documents")
DEFAULT_DOC_FILE = os.path.join(__parent_dir, "documents", "doc.docx")