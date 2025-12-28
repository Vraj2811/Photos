from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

Base = declarative_base()

# SQLAlchemy Models
class ImageRecord(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=True)  # Nullable for Drive images
    drive_file_id = Column(String(255), nullable=True)
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
