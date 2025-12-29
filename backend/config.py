from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "image_search.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"

VISION_MODEL = "llava"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
DRIVE_FOLDER_ID = "1GJ1Bl35jOKckFSoZb4b3Ube5VBxfKvH-"
SERVICE_ACCOUNT_DIR = PROJECT_ROOT / "Service Account Utility" / "accounts"
THUMBNAIL_CACHE_DIR = PROJECT_ROOT / "backend" / "thumbnails"
