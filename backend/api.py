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

try:
    import ollama
    import faiss
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as e:
    print(f"Missing packages: {e}")
    exit(1)

from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "image_search.db"
IMAGES_FOLDER = PROJECT_ROOT / "images"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"

VISION_MODEL = "llava"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Ensure directories exist
IMAGES_FOLDER.mkdir(exist_ok=True)
FAISS_INDEX_PATH.mkdir(exist_ok=True)

# Database setup
Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    description = Column(Text)
    embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

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

# Database helper
class ImageDB:
    def __init__(self):
        self.SessionLocal = SessionLocal
    
    def add_image(self, filename, file_path, description, embedding):
        session = self.SessionLocal()
        try:
            embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            image_record = ImageRecord(
                filename=filename,
                file_path=file_path,
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
    
    def generate_description(self, image_path):
        try:
            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in 2-3 clear, concise sentences.',
                    'images': [str(image_path)]
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
app.mount("/images", StaticFiles(directory=str(IMAGES_FOLDER)), name="images")

# API Endpoints

@app.get("/")
async def root():
    return {"message": "AI Image Search API", "version": "1.0.0"}

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    all_images = db.get_all_images()
    
    return SystemStatus(
        total_images=len(all_images),
        total_vectors=vector_db.index.ntotal if vector_db.index else 0,
        ollama_connected=len(ollama_proc.models_available) > 0,
        models_available=ollama_proc.models_available,
        status="ready" if len(all_images) > 0 else "empty"
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
                image_url=f"/images/{img.filename}"
            ))
    
    return results

@app.get("/api/images", response_model=List[ImageInfo])
async def get_all_images(limit: int = Query(50, le=200)):
    """Get all images"""
    all_images = db.get_all_images()
    
    results = []
    for img in all_images[:limit]:
        results.append(ImageInfo(
            id=img.id,
            filename=img.filename,
            description=img.description or "",
            created_at=img.created_at.isoformat() if img.created_at else "",
            image_url=f"/images/{img.filename}"
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
        image_url=f"/images/{img.filename}"
    )

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate filename
        timestamp = int(time.time())
        file_hash = hashlib.md5(contents).hexdigest()[:8]
        filename = f"{timestamp}_{file_hash}_{file.filename}"
        file_path = IMAGES_FOLDER / filename
        
        # Save image
        image.save(file_path)
        
        # Generate description
        description = ollama_proc.generate_description(file_path)
        if not description:
            return {"success": False, "error": "Failed to generate description"}
        
        # Generate embedding
        embedding = ollama_proc.generate_embedding(description)
        if embedding is None:
            return {"success": False, "error": "Failed to generate embedding"}
        
        # Save to database
        image_id = db.add_image(filename, str(file_path), description, embedding)
        if not image_id:
            return {"success": False, "error": "Failed to save to database"}
        
        # Add to vector index
        vector_db.add_vector(embedding, image_id)
        
        return {
            "success": True,
            "image_id": image_id,
            "filename": filename,
            "description": description,
            "image_url": f"/images/{filename}"
        }
        
    except Exception as e:
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Image Search API...")
    print(f"📁 Images: {IMAGES_FOLDER}")
    print(f"💾 Database: {DATABASE_PATH}")
    print(f"🔢 Vectors: {vector_db.index.ntotal if vector_db.index else 0}")
    uvicorn.run(app, host="0.0.0.0", port=8000)





