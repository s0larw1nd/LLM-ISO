import os

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen2.5-7B-Instruct-merged")
DEFAULT_EMBEDDINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-Embedding-0.6B")
DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db")
DEFAULT_DOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
DEFAULT_DOC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "doc.docx")