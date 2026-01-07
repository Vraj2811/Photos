
#!/usr/bin/env python3
import os
import sys
import time
import hashlib
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional
import io
from PIL import Image

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.config import (
    DATABASE_PATH, UPLOAD_DIR, THUMBNAIL_CACHE_DIR
)
from backend.database import ImageDB
from backend.processors import OllamaProcessor, FaceProcessor
from backend.vector_db import VectorDB
from backend.api import process_image_for_serving

# Initialize components
db = ImageDB()
ollama_proc = OllamaProcessor()
vector_db = VectorDB()
face_proc = FaceProcessor(db)

# Ensure directories exist
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_LOG = Path(__file__).parent / "processed_files.log"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

def get_processed_files():
    """Read the log of already processed absolute file paths."""
    if not PROCESSED_LOG.exists():
        return set()
    return set(PROCESSED_LOG.read_text().splitlines())

def log_processed_file(file_path: Path):
    """Append a successfully processed absolute file path to the log."""
    with open(PROCESSED_LOG, "a") as f:
        f.write(f"{file_path.absolute()}\n")

async def process_file(file_path: Path, processed_set: set):
    """Process a single image or video file."""
    abs_path = str(file_path.absolute())
    if abs_path in processed_set:
        return True, "Already processed (skipped)"

    ext = file_path.suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS
    is_image = ext in IMAGE_EXTENSIONS
    
    if not (is_image or is_video):
        return False, f"Skipping unsupported file type: {ext}"

    try:
        # Read file content
        contents = file_path.read_bytes()
        
        # Generate filename for uploads folder
        timestamp = int(time.time())
        file_hash = hashlib.md5(contents).hexdigest()[:8]
        new_filename = f"{timestamp}_{file_hash}{ext}"
        target_path = UPLOAD_DIR / new_filename

        # Copy file to uploads
        shutil.copy2(file_path, target_path)

        description = "" # Default to blank as per user preference
        embedding = None
        image_id = None

        if is_image:
            # AI Processing
            print(f"  - Generating description...")
            description = ollama_proc.generate_description(contents)
            if not description:
                description = "Image uploaded (AI description unavailable)"
            
            print(f"  - Generating embedding...")
            embedding = ollama_proc.generate_embedding(description)
            
            # Face Detection
            print(f"  - Detecting faces...")
            detected_faces = face_proc.detect_faces(contents)
            
            # Thumbnail generation
            print(f"  - Generating thumbnail...")
            thumb_bytes = process_image_for_serving(contents, True)
        else:
            detected_faces = []
            thumb_bytes = None

        # Save to DB
        print(f"  - Saving to database...")
        image_id = db.add_image(file_path.name, str(target_path), description, embedding)
        
        if not image_id:
            if target_path.exists():
                target_path.unlink()
            return False, "Failed to save to database"

        # Vector Index
        if embedding is not None:
            print(f"  - Adding to vector index...")
            vector_db.add_vector(embedding, image_id)

        # Save Faces
        if detected_faces:
            print(f"  - Linking {len(detected_faces)} faces...")
            face_proc.save_faces(detected_faces, image_id)

        # Save Thumbnail
        if thumb_bytes:
            thumbnail_path = THUMBNAIL_CACHE_DIR / f"{image_id}_thumb.jpg"
            thumbnail_path.write_bytes(thumb_bytes)

        # Log success
        log_processed_file(file_path)
        return True, f"Successfully processed {file_path.name}"

    except Exception as e:
        return False, f"Error processing {file_path.name}: {str(e)}"

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bulk_process.py <directory_path>")
        return

    source_dir = Path(sys.argv[1])
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        return

    processed_set = get_processed_files()
    files = [f for f in source_dir.iterdir() if f.is_file()]
    total_files = len(files)
    print(f"Found {total_files} files in {source_dir}")
    print(f"Already processed: {len(processed_set)} files")

    processed_count = 0
    failed_count = 0
    skipped_count = 0

    for i, file_path in enumerate(files):
        print(f"[{i+1}/{total_files}] Processing {file_path.name}...")
        success, message = await process_file(file_path, processed_set)
        if success:
            if "skipped" in message:
                skipped_count += 1
            else:
                processed_count += 1
            print(f"  ✅ {message}")
        else:
            failed_count += 1
            print(f"  ❌ {message}")

    print("\nProcessing Complete!")
    print(f"Total: {total_files}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    asyncio.run(main())
