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
from PIL import Image
import io
import asyncio

from config import (
    DATABASE_PATH, DRIVE_FOLDER_ID, SERVICE_ACCOUNT_DIR, THUMBNAIL_CACHE_DIR
)
from models import (
    SearchQuery, SearchResult, ImageInfo, SystemStatus, VectorStats,
    FaceGroupInfo
)
from database import ImageDB
from processors import OllamaProcessor, FaceProcessor
from vector_db import VectorDB
from drive_client import DriveClient

# Initialize components
db = ImageDB()
ollama_proc = OllamaProcessor()
vector_db = VectorDB()
face_proc = FaceProcessor(db)
drive_client = DriveClient(str(SERVICE_ACCOUNT_DIR))

# Ensure thumbnail cache directory exists
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
                image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}",
                thumbnail_url=f"/api/drive-image-thumbnail/{img.drive_file_id}" if img.drive_file_id else None
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
            image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}",
            thumbnail_url=f"/api/drive-image-thumbnail/{img.drive_file_id}" if img.drive_file_id else None
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
        image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}",
        thumbnail_url=f"/api/drive-image-thumbnail/{img.drive_file_id}" if img.drive_file_id else None
    )

@app.delete("/api/images/{image_id}")
async def delete_image(image_id: int):
    """Delete image from DB and Drive"""
    img = db.get_image_by_id(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Delete from Drive
    if img.drive_file_id:
        print(f"Deleting from Drive: {img.drive_file_id}")
        drive_client.delete_file(img.drive_file_id)
    
    # Delete from Vector DB
    vector_db.delete_vector(image_id)
    
    # Delete from DB
    if db.delete_image(image_id):
        return {"success": True, "message": "Image deleted"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete from database")

@app.post("/api/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process image or video"""
    try:
        # Read file content
        contents = await file.read()
        
        # Determine if it's a video
        content_type = file.content_type or ""
        is_video = content_type.startswith("video/")
        
        description = None
        embedding = None
        
        if is_video:
            print(f"Video detected: {file.filename}")
            description = "Video uploaded (AI processing skipped)"
        else:
            # Validate image if not video
            try:
                image = Image.open(io.BytesIO(contents))
                image.verify()
                # Re-open because verify() can close the file pointer or leave it at end
                image = Image.open(io.BytesIO(contents)) 
            except Exception:
                 raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Generate description for image
            print(f"Generating description for {file.filename}...")
            description = await run_in_threadpool(ollama_proc.generate_description, contents)
            
            if not description:
                print(f"Warning: Failed to generate description for {file.filename}. Using placeholder.")
                description = "Image uploaded (AI description unavailable due to system limitations)"
            else:
                # Generate embedding only if description succeeded
                print(f"Generating embedding for {file.filename}...")
                embedding = await run_in_threadpool(ollama_proc.generate_embedding, description)

        # Generate filename
        timestamp = int(time.time())
        file_hash = hashlib.md5(contents).hexdigest()[:8]
        filename = f"{timestamp}_{file_hash}_{file.filename}"

        # Upload to Drive
        print(f"Uploading {filename} to Drive...")
        drive_file_id = await run_in_threadpool(drive_client.upload_file, filename, contents, DRIVE_FOLDER_ID, content_type or 'application/octet-stream')
        
        if not drive_file_id:
             print(f"Upload failed for {filename}. Skipping database entry.")
             return {
                 "success": False, 
                 "error": "Failed to upload to Google Drive after multiple attempts. Image not saved."
             }
        
        # If it's a video, return early and do not add to database
        if is_video:
            print(f"Video {filename} uploaded to Drive. Skipping database entry as requested.")
            return {
                "success": True,
                "image_id": None,
                "filename": filename,
                "description": description,
                "image_url": f"/api/drive-image/{drive_file_id}",
                "message": "Video uploaded to Drive (not stored in database)"
            }

        # Save to database
        print(f"Saving {filename} to database...")
        image_id = db.add_image(filename, None, description, embedding, drive_file_id=drive_file_id)
        if not image_id:
            print(f"Failed to save to DB. Cleaning up Drive file {drive_file_id}...")
            drive_client.delete_file(drive_file_id)
            return {"success": False, "error": "Failed to save to database"}
        
        # Add to vector index if embedding exists
        if embedding is not None:
            vector_db.add_vector(embedding, image_id)
        
        # Process faces and generate thumbnail asynchronously
        print(f"Queueing background tasks for {filename}...")
        background_tasks.add_task(face_proc.process_image, contents, image_id)
        background_tasks.add_task(get_drive_image_thumbnail, drive_file_id)
        
        return {
            "success": True,
            "image_id": image_id,
            "filename": filename,
            "description": description,
            "image_url": f"/api/drive-image/{drive_file_id}"
        }
        
    except Exception as e:
        print(f"Upload failed: {e}")
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

@app.post("/api/sync-drive")
async def sync_drive(background_tasks: BackgroundTasks):
    """Sync images from Google Drive"""
    background_tasks.add_task(run_sync_drive, background_tasks)
    return {"success": True, "message": "Sync started in background"}

async def run_sync_drive(background_tasks: BackgroundTasks):
    """Internal function for background sync"""
    try:
        print(f"Syncing from Drive Folder: {DRIVE_FOLDER_ID}")
        files = await run_in_threadpool(drive_client.list_images_in_folder, DRIVE_FOLDER_ID)
        print(f"Found {len(files)} images in Drive")
        
        count = 0
        for file in files:
            file_id = file['id']
            filename = file['name']
            
            # Check if already exists
            if db.get_image_by_drive_id(file_id):
                continue
            
            print(f"Processing {filename} ({file_id})...")
            
            # Download content
            content = await run_in_threadpool(drive_client.download_file, file_id)
            
            # Generate description
            description = None
            embedding = None
            
            try:
                # Ollama accepts bytes directly in 'images' list
                description = await run_in_threadpool(ollama_proc.generate_description, content)
                if description:
                    # Generate embedding
                    embedding = await run_in_threadpool(ollama_proc.generate_embedding, description)
            except Exception as e:
                print(f"AI processing failed for {filename}: {e}")
                description = "Image from Google Drive (AI processing skipped)"
            
            # Save to DB even if AI failed
            image_id = db.add_image(filename, None, description, embedding, drive_file_id=file_id)
            if image_id:
                if embedding is not None:
                    vector_db.add_vector(embedding, image_id)
                
                # Process faces
                print(f"Processing faces for {filename}...")
                background_tasks.add_task(face_proc.process_image, content, image_id)
                
                # Pre-generate thumbnail
                print(f"Pre-generating thumbnail for {filename}...")
                await get_drive_image_thumbnail(file_id)
                
                count += 1
                print(f"Added {filename}")
        
        # After sync, ensure all images have thumbnails
        background_tasks.add_task(generate_all_missing_thumbnails)
        
        return {"success": True, "count": count}
            
    except Exception as e:
        print(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/drive-image/{file_id}")
async def get_drive_image(file_id: str):
    """Serve image from Google Drive"""
    try:
        content = await run_in_threadpool(drive_client.download_file, file_id)
        return Response(content=content, media_type="image/jpeg") 
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/api/drive-image-thumbnail/{file_id}")
async def get_drive_image_thumbnail(file_id: str):
    """Serve thumbnail from Google Drive (with local caching)"""
    thumbnail_path = THUMBNAIL_CACHE_DIR / f"{file_id}_thumb.jpg"
    
    if thumbnail_path.exists():
        return Response(content=thumbnail_path.read_bytes(), media_type="image/jpeg")
    
    try:
        # Download original
        content = await run_in_threadpool(drive_client.download_file, file_id)
        
        # Create thumbnail
        img = Image.open(io.BytesIO(content))
        # Convert to RGB if necessary (for PNG/RGBA)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        img.thumbnail((400, 400)) # Max size for gallery grid
        
        # Save to cache
        thumb_io = io.BytesIO()
        img.save(thumb_io, format="JPEG", quality=85)
        thumb_bytes = thumb_io.getvalue()
        
        thumbnail_path.write_bytes(thumb_bytes)
        
        return Response(content=thumb_bytes, media_type="image/jpeg")
    except Exception as e:
        print(f"Thumbnail generation failed for {file_id}: {e}")
        # If it's a 404, we should probably return a 404
        if "File not found" in str(e) or "404" in str(e):
             raise HTTPException(status_code=404, detail="Image not found on Drive")
             
        # Fallback to original if thumbnail fails for other reasons
        try:
            content = await run_in_threadpool(drive_client.download_file, file_id)
            return Response(content=content, media_type="image/jpeg")
        except:
            raise HTTPException(status_code=404, detail="Image not found")

async def generate_all_missing_thumbnails():
    """Background task to generate thumbnails for all images that don't have them."""
    print("Starting background thumbnail generation...")
    all_images = db.get_all_images()
    count = 0
    for img in all_images:
        if img.drive_file_id:
            thumbnail_path = THUMBNAIL_CACHE_DIR / f"{img.drive_file_id}_thumb.jpg"
            if not thumbnail_path.exists():
                try:
                    await get_drive_image_thumbnail(img.drive_file_id)
                    count += 1
                    if count % 10 == 0:
                        print(f"Generated {count} thumbnails...")
                except Exception as e:
                    print(f"Failed to generate thumbnail for {img.drive_file_id}: {e}")
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
            
        # Use the first face's image as representative
        rep_face = faces[0]
        img = db.get_image_by_id(rep_face.image_id)
        
        results.append(FaceGroupInfo(
            id=group.id,
            name=group.name,
            image_count=len(faces),
            representative_image_url=f"/api/drive-image-thumbnail/{img.drive_file_id}" if img and img.drive_file_id else None
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
                image_url=f"/api/drive-image/{img.drive_file_id}" if img and img.drive_file_id else f"/images/{img.filename}",
                thumbnail_url=f"/api/drive-image-thumbnail/{img.drive_file_id}" if img and img.drive_file_id else None
            ))
            
    return results

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Image Search API...")
    print(f"💾 Database: {DATABASE_PATH}")
    print(f"🔢 Vectors: {vector_db.index.ntotal if vector_db.index else 0}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
