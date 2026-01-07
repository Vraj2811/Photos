from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "image_search.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"

VISION_MODEL = "llava"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
THUMBNAIL_CACHE_DIR = PROJECT_ROOT / "backend" / "thumbnails"
