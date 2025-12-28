"""
FastAPI Backend for AI Image Search System
Provides REST API endpoints for React frontend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import time
import hashlib
import numpy as np
from PIL import Image
import io
import base64

try:
    import ollama
    import faiss
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as e:
    print(f"Missing packages: {e}")
    exit(1)

from datetime import datetime
from drive_client import DriveClient
from fastapi.responses import Response
import cv2
from insightface.app import FaceAnalysis
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

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

# Ensure directories exist

FAISS_INDEX_PATH.mkdir(exist_ok=True)

# Database setup
Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=True)  # Nullable for Drive images
    drive_file_id = Column(String(255), nullable=True) # New column
    description = Column(Text)
    embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    faces = relationship("DetectedFace", back_populates="image", cascade="all, delete-orphan")

class FaceGroup(Base):
    __tablename__ = 'face_groups'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)
    representative_embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    faces = relationship("DetectedFace", back_populates="group")

class DetectedFace(Base):
    __tablename__ = 'detected_faces'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    group_id = Column(Integer, ForeignKey('face_groups.id'))
    embedding = Column(Text)
    bbox = Column(Text)  # JSON string of [x1, y1, x2, y2]
    confidence = Column(Float)
    
    # Relationships
    image = relationship("ImageRecord", back_populates="faces")
    group = relationship("FaceGroup", back_populates="faces")

# Initialize database
engine = create_engine(DATABASE_URL, echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# Pydantic models for API
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    image_id: int
    filename: str
    description: str
    confidence: float
    image_url: str

class ImageInfo(BaseModel):
    id: int
    filename: str
    description: str
    created_at: str
    image_url: str

class SystemStatus(BaseModel):
    total_images: int
    total_vectors: int
    ollama_connected: bool
    models_available: List[str]
    status: str

class VectorStats(BaseModel):
    dimension: int
    norm: float
    mean: float
    std: float
    min_val: float
    max_val: float

class FaceGroupInfo(BaseModel):
    id: int
    name: Optional[str]
    image_count: int
    representative_image_url: Optional[str]

class DetectedFaceInfo(BaseModel):
    id: int
    image_id: int
    group_id: int
    bbox: List[float]
    confidence: float

# Database helper
class ImageDB:
    def __init__(self):
        self.SessionLocal = SessionLocal
    
    def add_image(self, filename, file_path, description, embedding, **kwargs):
        session = self.SessionLocal()
        try:
            if embedding is not None:
                embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            else:
                embedding_json = None
                
            image_record = ImageRecord(
                filename=filename,
                file_path=file_path,
                drive_file_id=kwargs.get('drive_file_id'),
                description=description,
                embedding=embedding_json,
                created_at=datetime.now()
            )
            session.add(image_record)
            session.commit()
            return image_record.id
        except Exception as e:
            session.rollback()
            print(f"Database error: {e}")
            return None
        finally:
            session.close()
    
    def get_all_images(self):
        session = self.SessionLocal()
        try:
            return session.query(ImageRecord).order_by(ImageRecord.created_at.desc()).all()
        finally:
            session.close()
    
    def get_image_by_id(self, image_id):
        session = self.SessionLocal()
        try:
            return session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        finally:
            session.close()

    def get_image_by_drive_id(self, drive_id):
        session = self.SessionLocal()
        try:
            return session.query(ImageRecord).filter(ImageRecord.drive_file_id == drive_id).first()
        finally:
            session.close()

    def delete_image(self, image_id):
        session = self.SessionLocal()
        try:
            image = session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            if image:
                session.delete(image)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Delete failed: {e}")
            return False
        finally:
            session.close()

    def add_face_group(self, embedding):
        session = self.SessionLocal()
        try:
            embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            group = FaceGroup(representative_embedding=embedding_json)
            session.add(group)
            session.commit()
            return group.id
        except Exception as e:
            session.rollback()
            print(f"Failed to add face group: {e}")
            return None
        finally:
            session.close()

    def add_detected_face(self, image_id, group_id, embedding, bbox, confidence):
        session = self.SessionLocal()
        try:
            embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            bbox_json = json.dumps(bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))
            face = DetectedFace(
                image_id=image_id,
                group_id=group_id,
                embedding=embedding_json,
                bbox=bbox_json,
                confidence=float(confidence)
            )
            session.add(face)
            session.commit()
            return face.id
        except Exception as e:
            session.rollback()
            print(f"Failed to add detected face: {e}")
            return None
        finally:
            session.close()

    def get_all_face_groups(self):
        session = self.SessionLocal()
        try:
            return session.query(FaceGroup).all()
        finally:
            session.close()

    def get_faces_by_group(self, group_id):
        session = self.SessionLocal()
        try:
            return session.query(DetectedFace).filter(DetectedFace.group_id == group_id).all()
        finally:
            session.close()

# Ollama processor
class OllamaProcessor:
    def __init__(self):
        self.vision_model = VISION_MODEL
        self.embedding_model = EMBEDDING_MODEL
        self.models_available = []
        self.check_models()
    
    def check_models(self):
        try:
            models_response = ollama.list()
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        self.models_available.append(model.model.split(':')[0])
            return True
        except:
            return False
    
    def generate_description(self, image_input):
        try:
            images = []
            if isinstance(image_input, bytes):
                # Convert bytes to base64 string
                base64_image = base64.b64encode(image_input).decode('utf-8')
                images.append(base64_image)
            else:
                # Path string
                images.append(image_input)

            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in 2-3 clear, concise sentences.',
                    'images': images
                }]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Description failed: {e}")
            return None
    
    def generate_embedding(self, text):
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            if 'embedding' in response and response['embedding']:
                embedding = np.array(response['embedding'], dtype=np.float32)
                
                if len(embedding) > EMBEDDING_DIMENSION:
                    embedding = embedding[:EMBEDDING_DIMENSION]
                elif len(embedding) < EMBEDDING_DIMENSION:
                    padded = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
            return None
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None

# Face Processor
class FaceProcessor:
    def __init__(self):
        print("Initializing FaceAnalysis(name='buffalo_l')...")
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.similarity_threshold = 0.6

    def process_image(self, image_bytes, image_id):
        try:
            # Convert bytes to cv2 image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Failed to decode image for face analysis (ID: {image_id})")
                return
            
            faces = self.app.get(img)
            print(f"Detected {len(faces)} faces in image {image_id}")
            
            for face in faces:
                embedding = face.normed_embedding
                bbox = face.bbox
                confidence = face.det_score
                
                # Find matching group
                group_id = self.find_matching_group(embedding)
                
                if group_id is None:
                    # Create new group
                    group_id = db.add_face_group(embedding)
                    print(f"Created new face group {group_id}")
                
                # Save detected face
                db.add_detected_face(image_id, group_id, embedding, bbox, confidence)
                
        except Exception as e:
            print(f"Face processing failed for image {image_id}: {e}")

    def find_matching_group(self, face_embedding):
        groups = db.get_all_face_groups()
        best_match_id = None
        best_similarity = -1
        
        for group in groups:
            group_embedding = np.array(json.loads(group.representative_embedding), dtype=np.float32)
            similarity = np.dot(face_embedding, group_embedding)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = group.id
                
        return best_match_id

# FAISS vector DB
class VectorDB:
    def __init__(self):
        self.index_file = FAISS_INDEX_PATH / "vectors.index"
        self.mapping_file = FAISS_INDEX_PATH / "mapping.json"
        self.index = None
        self.id_mapping = {}
        self.load_or_create_index()
    
    def load_or_create_index(self):
        if self.index_file.exists() and self.mapping_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    self.id_mapping = {int(k): v for k, v in mapping_data.items()}
                print(f"Loaded index with {self.index.ntotal} vectors")
            except:
                self.create_new_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.id_mapping = {}
    
    def add_vector(self, embedding, image_id):
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            self.index.add(embedding)
            faiss_idx = self.index.ntotal - 1
            self.id_mapping[faiss_idx] = image_id
            self.save_index()
            return True
        except Exception as e:
            print(f"Failed to add vector: {e}")
            return False
    
    def search(self, query_embedding, top_k=5):
        try:
            if self.index.ntotal == 0:
                return []
            
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and int(idx) in self.id_mapping:
                    results.append({
                        'image_id': self.id_mapping[int(idx)],
                        'confidence': float(score)
                    })
            return results
        except:
            return []
    
    def delete_vector(self, image_id):
        try:
            # Find faiss index for image_id
            faiss_idx = -1
            for k, v in self.id_mapping.items():
                if v == image_id:
                    faiss_idx = k
                    break
            
            if faiss_idx != -1:
                # Remove from FAISS
                self.index.remove_ids(np.array([faiss_idx], dtype=np.int64))
                
                # Update mapping (shift indices)
                new_mapping = {}
                for k, v in self.id_mapping.items():
                    if k < faiss_idx:
                        new_mapping[k] = v
                    elif k > faiss_idx:
                        new_mapping[k - 1] = v
                self.id_mapping = new_mapping
                
                self.save_index()
                print(f"Deleted vector for image {image_id} (index {faiss_idx})")
                return True
            return False
        except Exception as e:
            print(f"Failed to delete vector: {e}")
            return False

    def save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in self.id_mapping.items()}
                json.dump(mapping_str, f)
        except Exception as e:
            print(f"Save failed: {e}")

# Initialize components
db = ImageDB()
ollama_proc = OllamaProcessor()
vector_db = VectorDB()
face_proc = FaceProcessor()
drive_client = DriveClient(str(SERVICE_ACCOUNT_DIR))

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

# Serve images


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
    query_embedding = ollama_proc.generate_embedding(query.query)
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
                image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}"
            ))
    
    return results

@app.get("/api/images", response_model=List[ImageInfo])
async def get_all_images(limit: int = Query(50, le=200)):
    """Get all images (excluding videos)"""
    all_images = db.get_all_images()
    
    # Filter out videos
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    filtered_images = [
        img for img in all_images 
        if not any(img.filename.lower().endswith(ext) for ext in video_extensions)
    ]
    
    results = []
    for img in filtered_images[:limit]:
        results.append(ImageInfo(
            id=img.id,
            filename=img.filename,
            description=img.description or "",
            created_at=img.created_at.isoformat() if img.created_at else "",
            image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}"
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
        image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}"
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
async def upload_image(file: UploadFile = File(...)):
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
            description = ollama_proc.generate_description(contents)
            
            if not description:
                print(f"Warning: Failed to generate description for {file.filename}. Using placeholder.")
                description = "Image uploaded (AI description unavailable due to system limitations)"
            else:
                # Generate embedding only if description succeeded
                print(f"Generating embedding for {file.filename}...")
                embedding = ollama_proc.generate_embedding(description)

        # Generate filename
        timestamp = int(time.time())
        file_hash = hashlib.md5(contents).hexdigest()[:8]
        filename = f"{timestamp}_{file_hash}_{file.filename}"

        # Upload to Drive
        print(f"Uploading {filename} to Drive...")
        drive_file_id = drive_client.upload_file(filename, contents, DRIVE_FOLDER_ID, mime_type=content_type or 'application/octet-stream')
        
        if not drive_file_id:
             raise HTTPException(status_code=500, detail="Failed to upload to Drive")
        
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
        
        # Process faces asynchronously (simulated for now, could use BackgroundTasks)
        print(f"Processing faces for {filename}...")
        face_proc.process_image(contents, image_id)
        
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
async def sync_drive():
    """Sync images from Google Drive"""
    try:
        print(f"Syncing from Drive Folder: {DRIVE_FOLDER_ID}")
        files = drive_client.list_images_in_folder(DRIVE_FOLDER_ID)
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
            content = drive_client.download_file(file_id)
            
            # Generate description
            description = None
            embedding = None
            
            try:
                # Ollama accepts bytes directly in 'images' list
                description = ollama_proc.generate_description(content)
                if description:
                    # Generate embedding
                    embedding = ollama_proc.generate_embedding(description)
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
                face_proc.process_image(content, image_id)
                
                count += 1
                print(f"Added {filename}")
        
        return {"success": True, "count": count}
            
    except Exception as e:
        print(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/drive-image/{file_id}")
async def get_drive_image(file_id: str):
    """Serve image from Google Drive"""
    try:
        content = drive_client.download_file(file_id)
        return Response(content=content, media_type="image/jpeg") 
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found")

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
            representative_image_url=f"/api/drive-image/{img.drive_file_id}" if img and img.drive_file_id else None
        ))
        
    return results

@app.get("/api/face-groups/{group_id}", response_model=List[ImageInfo])
async def get_face_group_images(group_id: int):
    """Get all images in a face group"""
    faces = db.get_faces_by_group(group_id)
    if not faces:
        raise HTTPException(status_code=404, detail="Group not found or empty")
        
    image_ids = list(set(face.image_id for face in faces))
    results = []
    
    for img_id in image_ids:
        img = db.get_image_by_id(img_id)
        if img:
            results.append(ImageInfo(
                id=img.id,
                filename=img.filename,
                description=img.description or "",
                created_at=img.created_at.isoformat() if img.created_at else "",
                image_url=f"/api/drive-image/{img.drive_file_id}" if img.drive_file_id else f"/images/{img.filename}"
            ))
            
    return results

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Image Search API...")
    print(f"💾 Database: {DATABASE_PATH}")
    print(f"🔢 Vectors: {vector_db.index.ntotal if vector_db.index else 0}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
