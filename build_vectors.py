#!/usr/bin/env python3
"""
Build vectors for all images in the images folder.
This script will:
1. Scan the images folder
2. Generate descriptions using LLaVA
3. Create embeddings
4. Store in database and FAISS index
"""

from pathlib import Path
import json
import numpy as np
from datetime import datetime
import time

try:
    import ollama
    import faiss
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("Install with: pip install ollama-python faiss-cpu sqlalchemy")
    exit(1)

# Configuration
PROJECT_ROOT = Path(__file__).parent
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

def init_database():
    """Initialize database"""
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal

def check_ollama():
    """Check if Ollama is running and models are available"""
    try:
        models_response = ollama.list()
        available = []
        
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                if hasattr(model, 'model'):
                    available.append(model.model.split(':')[0])
        
        print(f"✓ Ollama connected. Available models: {', '.join(available)}")
        
        if VISION_MODEL not in available:
            print(f"❌ {VISION_MODEL} not found. Install with: ollama pull {VISION_MODEL}")
            return False
        
        if EMBEDDING_MODEL not in available:
            print(f"❌ {EMBEDDING_MODEL} not found. Install with: ollama pull {EMBEDDING_MODEL}")
            return False
        
        print(f"✓ Both required models available")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False

def generate_description(image_path):
    """Generate description using LLaVA"""
    try:
        print(f"  → Generating description...")
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Describe this image in 2-3 clear, concise sentences. Focus on the main subject, key details, and overall scene.',
                'images': [str(image_path)]
            }]
        )
        desc = response['message']['content'].strip()
        print(f"  ✓ Description: {desc[:100]}...")
        return desc
    except Exception as e:
        print(f"  ❌ Description failed: {e}")
        return None

def generate_embedding(text):
    """Generate embedding from text"""
    try:
        print(f"  → Generating embedding...")
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        
        if 'embedding' in response and response['embedding']:
            embedding = np.array(response['embedding'], dtype=np.float32)
            
            # Ensure correct dimension
            if len(embedding) > EMBEDDING_DIMENSION:
                embedding = embedding[:EMBEDDING_DIMENSION]
            elif len(embedding) < EMBEDDING_DIMENSION:
                padded = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            print(f"  ✓ Embedding generated ({len(embedding)} dimensions)")
            return embedding
        else:
            print(f"  ❌ Invalid embedding response")
            return None
    except Exception as e:
        print(f"  ❌ Embedding failed: {e}")
        return None

def process_images():
    """Main processing function"""
    print("=" * 70)
    print("🔨 Image Vector Builder")
    print("=" * 70)
    
    # Check Ollama
    if not check_ollama():
        print("\n❌ Please fix Ollama setup before continuing.")
        return
    
    print()
    
    # Initialize database
    print("→ Initializing database...")
    SessionLocal = init_database()
    session = SessionLocal()
    
    # Get existing images in database
    existing_images = session.query(ImageRecord).all()
    existing_filenames = {img.filename for img in existing_images}
    print(f"✓ Database has {len(existing_images)} existing images")
    
    # Scan images folder
    print(f"→ Scanning {IMAGES_FOLDER}...")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        image_files.extend(IMAGES_FOLDER.glob(ext))
        image_files.extend(IMAGES_FOLDER.glob(ext.upper()))
    
    print(f"✓ Found {len(image_files)} image files")
    
    # Filter out already processed images
    new_images = [img for img in image_files if img.name not in existing_filenames]
    
    if not new_images:
        print("\n✅ All images are already in the database!")
        print("→ Now rebuilding FAISS index...")
        rebuild_faiss_index(session)
        session.close()
        return
    
    print(f"→ Processing {len(new_images)} new images...")
    print()
    
    # Process each new image
    success_count = 0
    
    for i, image_path in enumerate(new_images, 1):
        print(f"\n[{i}/{len(new_images)}] Processing: {image_path.name}")
        
        # Generate description
        description = generate_description(image_path)
        if not description:
            print(f"  ❌ Skipping due to description error")
            continue
        
        # Generate embedding
        embedding = generate_embedding(description)
        if embedding is None:
            print(f"  ❌ Skipping due to embedding error")
            continue
        
        # Store in database
        try:
            embedding_json = json.dumps(embedding.tolist())
            
            image_record = ImageRecord(
                filename=image_path.name,
                file_path=str(image_path),
                description=description,
                embedding=embedding_json,
                created_at=datetime.now()
            )
            
            session.add(image_record)
            session.commit()
            print(f"  ✓ Saved to database (ID: {image_record.id})")
            success_count += 1
            
        except Exception as e:
            session.rollback()
            print(f"  ❌ Database error: {e}")
    
    session.close()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✅ Processing complete!")
    print(f"   Successfully processed: {success_count}/{len(new_images)} images")
    print("=" * 70)
    
    # Rebuild FAISS index
    if success_count > 0:
        print("\n→ Building FAISS index...")
        session = SessionLocal()
        rebuild_faiss_index(session)
        session.close()

def rebuild_faiss_index(session):
    """Rebuild FAISS index from database"""
    try:
        # Get all images from database
        all_images = session.query(ImageRecord).all()
        
        if not all_images:
            print("  ⚠️  No images in database to index")
            return
        
        print(f"  → Indexing {len(all_images)} images...")
        
        # Create new FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        id_mapping = {}
        
        # Add vectors
        vectors_to_add = []
        image_ids = []
        
        for img in all_images:
            if img.embedding:
                try:
                    embedding_data = json.loads(img.embedding)
                    embedding = np.array(embedding_data, dtype=np.float32)
                    
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    vectors_to_add.append(embedding)
                    image_ids.append(img.id)
                except Exception as e:
                    print(f"  ⚠️  Error loading embedding for image {img.id}: {e}")
        
        if vectors_to_add:
            vectors_array = np.vstack(vectors_to_add).astype(np.float32)
            index.add(vectors_array)
            
            # Create mapping
            for idx, image_id in enumerate(image_ids):
                id_mapping[idx] = image_id
            
            # Save to disk
            index_file = FAISS_INDEX_PATH / "vectors.index"
            mapping_file = FAISS_INDEX_PATH / "mapping.json"
            
            faiss.write_index(index, str(index_file))
            
            with open(mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in id_mapping.items()}
                json.dump(mapping_str, f, indent=2)
            
            print(f"  ✓ FAISS index built with {len(vectors_to_add)} vectors")
            print(f"  ✓ Saved to {index_file}")
        else:
            print("  ⚠️  No valid embeddings to index")
            
    except Exception as e:
        print(f"  ❌ FAISS rebuild failed: {e}")

if __name__ == "__main__":
    print("\n📸 Image Vector Builder for AI Search System\n")
    
    # Check if images folder exists
    if not IMAGES_FOLDER.exists():
        print(f"❌ Images folder not found: {IMAGES_FOLDER}")
        exit(1)
    
    # Run processing
    try:
        process_images()
        
        print("\n" + "=" * 70)
        print("🎉 All done! Your images are now searchable.")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Go to the Search tab")
        print("3. Start searching!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

