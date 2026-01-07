"""
FastAPI Backend for AI Image Search System
Provides REST API endpoints for React frontend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from typing import List
import json
import time
import hashlib
import numpy as np
from PIL import Image, ImageOps
import io
import asyncio
from pathlib import Path

from config import (
    DATABASE_PATH, UPLOAD_DIR, THUMBNAIL_CACHE_DIR
)
from models import (
    SearchQuery, SearchResult, ImageInfo, SystemStatus, VectorStats,
    FaceGroupInfo
)
from database import ImageDB
from processors import OllamaProcessor, FaceProcessor
from vector_db import VectorDB

# Initialize components
db = ImageDB()
ollama_proc = OllamaProcessor()
vector_db = VectorDB()
face_proc = FaceProcessor(db)

# Ensure directories exist
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="AI Image Search API",
    description="REST API for AI-powered image search",
    version="1.0.0"
)

# CORS middleware for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/")
async def root():
    return {"message": "AI Image Search API", "version": "1.0.0"}

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    all_items = db.get_all_images()
    
    # Filter out videos for the count
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    image_count = sum(1 for img in all_items if not any(img.filename.lower().endswith(ext) for ext in video_extensions))
    
    return SystemStatus(
        total_images=image_count,
        total_vectors=vector_db.index.ntotal if vector_db.index else 0,
        ollama_connected=len(ollama_proc.models_available) > 0,
        models_available=ollama_proc.models_available,
        status="ready" if image_count > 0 else "empty"
    )

@app.post("/api/search", response_model=List[SearchResult])
async def search_images(query: SearchQuery):
    """Search for images"""
    if vector_db.index.ntotal == 0:
        raise HTTPException(status_code=404, detail="No images in database")
    
    # Generate query embedding
    query_embedding = await run_in_threadpool(ollama_proc.generate_embedding, query.query)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    
    # Search
    vector_results = vector_db.search(query_embedding, query.top_k)
    
    # Get image details
    results = []
    for result in vector_results:
        img = db.get_image_by_id(result['image_id'])
        if img:
            results.append(SearchResult(
                image_id=img.id,
                filename=img.filename,
                description=img.description or "",
                confidence=result['confidence'],
                image_url=f"/api/image/{img.id}",
                thumbnail_url=f"/api/thumbnail/{img.id}"
            ))
    
    return results

@app.get("/api/images", response_model=List[ImageInfo])
async def get_all_images(limit: int = Query(50, le=200), offset: int = Query(0, ge=0)):
    """Get all images (excluding videos) with pagination"""
    # Note: Filtering videos in memory after fetching from DB is inefficient for large datasets.
    # However, for now we'll keep the logic and just add pagination to the DB call.
    # A better way would be to add a 'type' column to the database.
    
    all_images = db.get_all_images(limit=limit, offset=offset)
    
    results = []
    for img in all_images:
        # Filter out videos
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        if any(img.filename.lower().endswith(ext) for ext in video_extensions):
            continue
            
        results.append(ImageInfo(
            id=img.id,
            filename=img.filename,
            description=img.description or "",
            created_at=img.created_at.isoformat() if img.created_at else "",
            image_url=f"/api/image/{img.id}",
            thumbnail_url=f"/api/thumbnail/{img.id}"
        ))
    
    return results

@app.get("/api/images/{image_id}", response_model=ImageInfo)
async def get_image(image_id: int):
    """Get specific image"""
    img = db.get_image_by_id(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return ImageInfo(
        id=img.id,
        filename=img.filename,
        description=img.description or "",
        created_at=img.created_at.isoformat() if img.created_at else "",
        image_url=f"/api/image/{img.id}",
        thumbnail_url=f"/api/thumbnail/{img.id}"
    )

@app.delete("/api/images/{image_id}")
async def delete_image(image_id: int):
    """Delete image from DB and local storage"""
    img = db.get_image_by_id(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Delete local file
    if img.file_path:
        file_path = Path(img.file_path)
        if file_path.exists():
            print(f"Deleting local file: {file_path}")
            file_path.unlink()
    
    # Delete thumbnail
    thumbnail_path = THUMBNAIL_CACHE_DIR / f"{image_id}_thumb.jpg"
    if thumbnail_path.exists():
        thumbnail_path.unlink()

    # Delete from Vector DB
    vector_db.delete_vector(image_id)
    
    # Delete from DB
    if db.delete_image(image_id):
        return {"success": True, "message": "Image deleted"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete from database")

@app.post("/api/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process image or video with FULL parallel execution and rollback"""
    file_path = None
    image_id = None
    thumbnail_path = None
    try:
        # Read file content
        contents = await file.read()
        
        # Determine if it's a video
        content_type = file.content_type or ""
        is_video = content_type.startswith("video/")
        
        # Generate filename
        timestamp = int(time.time())
        file_hash = hashlib.md5(contents).hexdigest()[:8]
        ext = Path(file.filename).suffix
        filename = f"{timestamp}_{file_hash}{ext}"
        file_path = UPLOAD_DIR / filename

        # --- Define Parallel Tasks ---

        async def save_file_task():
            print(f"Saving {filename} to local storage...")
            def _save():
                with open(file_path, "wb") as f:
                    f.write(contents)
            await run_in_threadpool(_save)

        async def process_ai_task():
            if is_video:
                return "Video uploaded (AI processing skipped)", None
            
            # Validate image
            try:
                image = Image.open(io.BytesIO(contents))
                image.verify()
            except Exception:
                 raise HTTPException(status_code=400, detail="Invalid image file")
            
            print(f"Generating description for {file.filename}...")
            desc = await run_in_threadpool(ollama_proc.generate_description, contents)
            
            if not desc:
                print(f"Warning: AI description failed for {file.filename}")
                desc = "Image uploaded (AI description unavailable)"
                return desc, None
            
            print(f"Generating embedding for {file.filename}...")
            emb = await run_in_threadpool(ollama_proc.generate_embedding, desc)
            return desc, emb

        async def detect_faces_task():
            if is_video:
                return []
            print(f"Detecting faces in {file.filename}...")
            return await run_in_threadpool(face_proc.detect_faces, contents)

        async def generate_thumbnail_bytes_task():
            if is_video:
                return None
            print(f"Generating thumbnail bytes for {file.filename}...")
            return await run_in_threadpool(process_image_for_serving, contents, True)

        # --- Execute All Tasks Simultaneously ---
        print(f"🚀 Starting FULL parallel processing for {file.filename}...")
        results = await asyncio.gather(
            save_file_task(),
            process_ai_task(),
            detect_faces_task(),
            generate_thumbnail_bytes_task()
        )
        
        _, ai_results, detected_faces, thumb_bytes = results
        description, embedding = ai_results

        # If it's a video, return early
        if is_video:
            return {
                "success": True,
                "image_id": None,
                "filename": filename,
                "description": description,
                "image_url": f"/api/image/video/{filename}",
                "message": "Video uploaded locally"
            }

        # --- Save to Database ---
        print(f"Saving {filename} to database...")
        image_id = db.add_image(file.filename, str(file_path), description, embedding)
        if not image_id:
            raise Exception("Failed to save image to database")
        
        # --- Post-DB Parallel Tasks (Linking) ---
        
        # 1. Add to vector index
        if embedding is not None:
            print(f"Adding {filename} to vector index...")
            success = vector_db.add_vector(embedding, image_id)
            if not success:
                raise Exception("Failed to add image to vector index")
        
        # 2. Save detected faces
        if detected_faces:
            print(f"Linking {len(detected_faces)} faces to image {image_id}...")
            await run_in_threadpool(face_proc.save_faces, detected_faces, image_id)
        
        # 3. Save thumbnail to disk
        if thumb_bytes:
            thumbnail_path = THUMBNAIL_CACHE_DIR / f"{image_id}_thumb.jpg"
            print(f"Saving thumbnail to {thumbnail_path}...")
            def _save_thumb():
                thumbnail_path.write_bytes(thumb_bytes)
            await run_in_threadpool(_save_thumb)
        
        return {
            "success": True,
            "image_id": image_id,
            "filename": file.filename,
            "description": description,
            "image_url": f"/api/image/{image_id}",
            "thumbnail_url": f"/api/thumbnail/{image_id}"
        }
        
    except Exception as e:
        print(f"❌ Upload failed for {file.filename}: {e}")
        # Rollback: Delete local file
        if file_path and file_path.exists():
            print(f"Rolling back: Deleting local file {file_path}")
            file_path.unlink()
        
        # Rollback: Delete thumbnail
        if thumbnail_path and thumbnail_path.exists():
            print(f"Rolling back: Deleting thumbnail {thumbnail_path}")
            thumbnail_path.unlink()
        
        # Rollback: Delete DB entry and Vector
        if image_id:
            print(f"Rolling back: Deleting database entry {image_id}")
            db.delete_image(image_id)
            vector_db.delete_vector(image_id)
            
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector/{image_id}", response_model=VectorStats)
async def get_vector_stats(image_id: int):
    """Get vector statistics for image"""
    img = db.get_image_by_id(image_id)
    if not img or not img.embedding:
        raise HTTPException(status_code=404, detail="Vector not found")
    
    try:
        embedding_data = json.loads(img.embedding)
        vector = np.array(embedding_data, dtype=np.float32)
        
        return VectorStats(
            dimension=len(vector),
            norm=float(np.linalg.norm(vector)),
            mean=float(vector.mean()),
            std=float(vector.std()),
            min_val=float(vector.min()),
            max_val=float(vector.max())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rebuild-index")
async def rebuild_index():
    """Rebuild FAISS index from database"""
    try:
        all_images = db.get_all_images()
        if not all_images:
            return {"success": False, "message": "No images to index"}
        
        vector_db.create_new_index()
        count = 0
        
        for img in all_images:
            if img.embedding:
                try:
                    embedding_data = json.loads(img.embedding)
                    embedding = np.array(embedding_data, dtype=np.float32)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    if vector_db.add_vector(embedding, img.id):
                        count += 1
                except:
                    continue
        
        return {"success": True, "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Removed sync-drive endpoint as it's no longer needed for local storage.

def process_image_for_serving(content: bytes, is_thumbnail: bool = False) -> bytes:
    """Process image bytes to fix orientation, color space, and optionally resize."""
    try:
        img = Image.open(io.BytesIO(content))
        original_mode = img.mode
        
        # 1. Fix orientation based on EXIF
        img = ImageOps.exif_transpose(img)
        
        # 2. Handle color space
        if original_mode == "CMYK":
            # CMYK JPEGs often appear inverted when converted directly to RGB
            # A common fix is to convert to RGB and then invert
            img = img.convert("RGB")
            img = ImageOps.invert(img)
        elif img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
            
        # 3. Resize if thumbnail
        if is_thumbnail:
            img.thumbnail((400, 400))
            
        # 4. Save to bytes
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG", quality=85 if is_thumbnail else 95)
        return img_io.getvalue()
    except Exception as e:
        print(f"Image processing failed: {e}")
        return content # Fallback to original content

def crop_face(image_bytes: bytes, bbox: List[float]) -> bytes:
    """Crop face from image bytes using bounding box [x1, y1, x2, y2]."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img) # Fix orientation before cropping
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Bounding box is [x1, y1, x2, y2]
        # Add some padding (20%)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        padding_w = w * 0.2
        padding_h = h * 0.2
        
        # New coordinates with padding
        nx1 = max(0, x1 - padding_w)
        ny1 = max(0, y1 - padding_h)
        nx2 = min(img.width, x2 + padding_w)
        ny2 = min(img.height, y2 + padding_h)
        
        face_img = img.crop((nx1, ny1, nx2, ny2))
        
        # Resize to a standard size for profile photos
        face_img.thumbnail((300, 300))
        
        img_io = io.BytesIO()
        face_img.save(img_io, format="JPEG", quality=90)
        return img_io.getvalue()
    except Exception as e:
        print(f"Face cropping failed: {e}")
        return image_bytes

@app.get("/api/face-thumbnail/{face_id}")
async def get_face_thumbnail(face_id: int):
    """Serve zoomed-in face thumbnail"""
    thumbnail_path = THUMBNAIL_CACHE_DIR / f"face_{face_id}.jpg"
    
    if thumbnail_path.exists():
        return Response(content=thumbnail_path.read_bytes(), media_type="image/jpeg")
    
    try:
        face = db.get_face_by_id(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Face record not found")
            
        img_record = db.get_image_by_id(face.image_id)
        if not img_record or not img_record.file_path:
            raise HTTPException(status_code=404, detail="Original image not found")
            
        # Read original
        file_path = Path(img_record.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        content = file_path.read_bytes()
        
        # Parse bbox
        bbox = json.loads(face.bbox)
        
        # Crop face
        face_bytes = await run_in_threadpool(crop_face, content, bbox)
        
        # Save to cache
        thumbnail_path.write_bytes(face_bytes)
        
        return Response(content=face_bytes, media_type="image/jpeg")
    except Exception as e:
        print(f"Face thumbnail generation failed for {face_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/{image_id}")
async def get_image_file(image_id: int):
    """Serve image from local storage"""
    img = db.get_image_by_id(image_id)
    if not img or not img.file_path:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        file_path = Path(img.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
            
        content = file_path.read_bytes()
        processed_content = await run_in_threadpool(process_image_for_serving, content)
        return Response(content=processed_content, media_type="image/jpeg") 
    except Exception as e:
        print(f"Error serving image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/video/{filename}")
async def get_video_file(filename: str):
    """Serve video file from local storage"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return Response(content=file_path.read_bytes(), media_type="video/mp4")

@app.get("/api/thumbnail/{image_id}")
async def get_thumbnail(image_id: int):
    """Serve thumbnail (with local caching)"""
    thumbnail_path = THUMBNAIL_CACHE_DIR / f"{image_id}_thumb.jpg"
    
    if thumbnail_path.exists():
        return Response(content=thumbnail_path.read_bytes(), media_type="image/jpeg")
    
    return await generate_thumbnail(image_id)

async def generate_thumbnail(image_id: int):
    """Generate and save thumbnail for an image"""
    img = db.get_image_by_id(image_id)
    if not img or not img.file_path:
        raise HTTPException(status_code=404, detail="Image not found")
        
    thumbnail_path = THUMBNAIL_CACHE_DIR / f"{image_id}_thumb.jpg"
    
    try:
        file_path = Path(img.file_path)
        if not file_path.exists():
             raise HTTPException(status_code=404, detail="Original file not found")
             
        content = file_path.read_bytes()
        thumb_bytes = await run_in_threadpool(process_image_for_serving, content, True)
        
        # Save to cache
        thumbnail_path.write_bytes(thumb_bytes)
        
        return Response(content=thumb_bytes, media_type="image/jpeg")
    except Exception as e:
        print(f"Thumbnail generation failed for {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_all_missing_thumbnails():
    """Background task to generate thumbnails for all images that don't have them."""
    print("Starting background thumbnail generation...")
    all_images = db.get_all_images()
    count = 0
    for img in all_images:
        thumbnail_path = THUMBNAIL_CACHE_DIR / f"{img.id}_thumb.jpg"
        if not thumbnail_path.exists():
            try:
                await generate_thumbnail(img.id)
                count += 1
                if count % 10 == 0:
                    print(f"Generated {count} thumbnails...")
            except Exception as e:
                print(f"Failed to generate thumbnail for {img.id}: {e}")
    print(f"Finished background thumbnail generation. Generated {count} thumbnails.")

@app.on_event("startup")
async def startup_event():
    """Run tasks on startup"""
    # Start background thumbnail generation
    asyncio.create_task(generate_all_missing_thumbnails())

@app.get("/api/face-groups", response_model=List[FaceGroupInfo])
async def get_face_groups():
    """Get all face groups with representative images"""
    groups = db.get_all_face_groups()
    results = []
    
    for group in groups:
        faces = db.get_faces_by_group(group.id)
        if not faces:
            continue
            
        # Use the first face's ID to generate a zoomed face thumbnail
        rep_face = faces[0]
        
        results.append(FaceGroupInfo(
            id=group.id,
            name=group.name,
            image_count=len(faces),
            representative_image_url=f"/api/face-thumbnail/{rep_face.id}"
        ))
        
    return results

@app.get("/api/face-groups/{group_id}", response_model=List[ImageInfo])
async def get_face_group_images(group_id: int, limit: int = Query(50, le=200), offset: int = Query(0, ge=0)):
    """Get all images in a face group with pagination"""
    faces = db.get_faces_by_group(group_id)
    if not faces:
        raise HTTPException(status_code=404, detail="Group not found or empty")
        
    image_ids = sorted(list(set(face.image_id for face in faces)), reverse=True)
    
    # Apply pagination to image_ids
    paginated_ids = image_ids[offset : offset + limit]
    
    results = []
    for img_id in paginated_ids:
        img = db.get_image_by_id(img_id)
        if img:
            results.append(ImageInfo(
                id=img.id,
                filename=img.filename,
                description=img.description or "",
                created_at=img.created_at.isoformat() if img.created_at else "",
                image_url=f"/api/image/{img.id}",
                thumbnail_url=f"/api/thumbnail/{img.id}"
            ))
            
    return results

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Image Search API...")
    print(f"💾 Database: {DATABASE_PATH}")
    print(f"🔢 Vectors: {vector_db.index.ntotal if vector_db.index else 0}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
