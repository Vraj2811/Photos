#!/usr/bin/env python3
"""
AI Image Search System - Gradio Version
Search images using natural language with LLaVA and FAISS
"""

import gradio as gr
from pathlib import Path
import json
import time
from PIL import Image
import numpy as np
from datetime import datetime
import hashlib
import traceback

try:
    import ollama as ollama_client
    import faiss
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("Install with: pip install gradio ollama-python faiss-cpu pillow numpy sqlalchemy")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    PROJECT_ROOT = Path(__file__).parent
    DATABASE_PATH = PROJECT_ROOT / "image_search.db"
    IMAGES_FOLDER = PROJECT_ROOT / "images"
    FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"
    
    VISION_MODEL = "llava"
    EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_DIMENSION = 768
    
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

Config.IMAGES_FOLDER.mkdir(exist_ok=True)
Config.FAISS_INDEX_PATH.mkdir(exist_ok=True)

# ============================================================================
# DATABASE SETUP
# ============================================================================

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
    engine = create_engine(Config.DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class ImageDB:
    def __init__(self, session_factory):
        self.SessionLocal = session_factory
    
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

# ============================================================================
# OLLAMA INTEGRATION
# ============================================================================

class OllamaProcessor:
    def __init__(self):
        self.vision_model = Config.VISION_MODEL
        self.embedding_model = Config.EMBEDDING_MODEL
        self.models_available = []
        self.is_connected = self.check_models()
    
    def check_models(self):
        try:
            models_response = ollama_client.list()
            available = []
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        available.append(model.model.split(':')[0])
            self.models_available = available
            return len(available) > 0
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False
    
    def generate_description(self, image_path):
        try:
            response = ollama_client.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in 2-3 clear, concise sentences. Focus on the main subject, key details, and overall scene.',
                    'images': [str(image_path)]
                }]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Description failed: {e}")
            return None
    
    def generate_embedding(self, text):
        """Generate embedding from text with detailed error reporting"""
        try:
            if not text or not text.strip():
                print("Error: Empty text provided for embedding")
                return None
            
            print(f"[DEBUG] Generating embedding for: '{text[:50]}...'")
            print(f"[DEBUG] Using model: {self.embedding_model}")
            
            response = ollama_client.embeddings(model=self.embedding_model, prompt=text)
            
            print(f"[DEBUG] Response received, type: {type(response)}")
            
            if 'embedding' in response and response['embedding']:
                embedding = np.array(response['embedding'], dtype=np.float32)
                print(f"[DEBUG] Embedding generated: {len(embedding)} dimensions")
                
                if len(embedding) > Config.EMBEDDING_DIMENSION:
                    embedding = embedding[:Config.EMBEDDING_DIMENSION]
                elif len(embedding) < Config.EMBEDDING_DIMENSION:
                    padded = np.zeros(Config.EMBEDDING_DIMENSION, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                print(f"[DEBUG] Final embedding: {len(embedding)} dims, norm={norm:.4f}")
                return embedding
            else:
                print(f"[ERROR] No embedding in response!")
                print(f"[ERROR] Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                return None
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# FAISS VECTOR DATABASE
# ============================================================================

class VectorDB:
    def __init__(self):
        self.index_file = Config.FAISS_INDEX_PATH / "vectors.index"
        self.mapping_file = Config.FAISS_INDEX_PATH / "mapping.json"
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
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                print("Creating new index...")
                self.create_new_index()
        else:
            print("No existing index found, creating new one...")
            self.create_new_index()
    
    def create_new_index(self):
        self.index = faiss.IndexFlatIP(Config.EMBEDDING_DIMENSION)
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
            print(f"  Embedding shape: {embedding.shape if hasattr(embedding, 'shape') else 'unknown'}")
            print(f"  Image ID: {image_id}")
            traceback.print_exc()
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
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    def save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in self.id_mapping.items()}
                json.dump(mapping_str, f, indent=2)
        except Exception as e:
            print(f"Failed to save index: {e}")
    
    def rebuild_from_database(self, db, ollama_processor):
        try:
            all_images = db.get_all_images()
            if not all_images:
                print("No images in database to rebuild index")
                return 0
            
            print(f"Rebuilding index for {len(all_images)} images...")
            self.create_new_index()
            count = 0
            errors = 0
            
            for img in all_images:
                embedding = None
                
                # Try to load existing embedding
                if img.embedding:
                    try:
                        embedding_data = json.loads(img.embedding)
                        embedding = np.array(embedding_data, dtype=np.float32)
                        
                        # Validate embedding dimension
                        if len(embedding) != Config.EMBEDDING_DIMENSION:
                            print(f"  Warning: Image {img.id} has wrong embedding dimension: {len(embedding)} != {Config.EMBEDDING_DIMENSION}")
                            embedding = None
                        else:
                            # Normalize
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                    except Exception as e:
                        print(f"  Error loading embedding for image {img.id}: {e}")
                        embedding = None
                
                # Generate embedding if missing or invalid
                if embedding is None and img.description:
                    print(f"  Generating new embedding for image {img.id}...")
                    embedding = ollama_processor.generate_embedding(img.description)
                
                # Add to index
                if embedding is not None:
                    if self.add_vector(embedding, img.id):
                        count += 1
                    else:
                        errors += 1
                        print(f"  Failed to add image {img.id} to index")
                else:
                    errors += 1
                    print(f"  Skipping image {img.id} - no valid embedding")
            
            print(f"Rebuild complete: {count} successful, {errors} errors")
            return count
        except Exception as e:
            print(f"Rebuild failed: {e}")
            traceback.print_exc()
            return 0

# ============================================================================
# IMAGE PROCESSING PIPELINE
# ============================================================================

class ImagePipeline:
    def __init__(self, db, ollama_processor, vector_db):
        self.db = db
        self.ollama = ollama_processor
        self.vector_db = vector_db
    
    def process_uploaded_image(self, image_file):
        try:
            # Get the file path from Gradio's upload
            if image_file is None:
                return False, "No image provided"
            
            # Read the image
            img = Image.open(image_file)
            
            # Generate unique filename
            timestamp = int(time.time())
            original_name = Path(image_file).name
            file_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
            filename = f"{timestamp}_{file_hash}_{original_name}"
            file_path = Config.IMAGES_FOLDER / filename
            
            # Save image
            img.save(file_path)
            
            # Generate description
            description = self.ollama.generate_description(file_path)
            if not description:
                return False, "Failed to generate description"
            
            # Generate embedding
            embedding = self.ollama.generate_embedding(description)
            if embedding is None:
                return False, "Failed to generate embedding"
            
            # Store in database
            image_id = self.db.add_image(filename, str(file_path), description, embedding)
            if not image_id:
                return False, "Failed to save to database"
            
            # Add to vector database
            self.vector_db.add_vector(embedding, image_id)
            
            return True, {
                'filename': filename,
                'description': description,
                'image_id': image_id,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def search_images(self, query, top_k=5):
        """Search for images similar to query"""
        try:
            print(f"\n[SEARCH] Query: '{query}'")
            
            if self.vector_db.index.ntotal == 0:
                return [], "No images in database. Please upload images first."
            
            print(f"[SEARCH] Index has {self.vector_db.index.ntotal} vectors")
            
            query_embedding = self.ollama.generate_embedding(query)
            if query_embedding is None:
                error_msg = """❌ Failed to generate embedding for query.

**Check terminal for detailed error messages.**

**Possible fixes:**
1. Make sure Ollama is running: `ollama serve`
2. Verify model is installed: `ollama list | grep nomic-embed-text`
3. If missing, install: `ollama pull nomic-embed-text`
"""
                return [], error_msg
            
            print(f"[SEARCH] Query embedding generated successfully")
            vector_results = self.vector_db.search(query_embedding, top_k)
            
            if not vector_results:
                return [], "No matching results found"
            
            results = []
            for result in vector_results:
                img = self.db.get_image_by_id(result['image_id'])
                if img:
                    results.append({
                        'image_id': img.id,
                        'filename': img.filename,
                        'file_path': img.file_path,
                        'description': img.description,
                        'confidence': result['confidence']
                    })
            
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results, None
            
        except Exception as e:
            return [], f"Search error: {str(e)}"

# ============================================================================
# INITIALIZE SYSTEM
# ============================================================================

SessionLocal = init_database()
db = ImageDB(SessionLocal)
ollama = OllamaProcessor()
vector_db = VectorDB()
pipeline = ImagePipeline(db, ollama, vector_db)

# ============================================================================
# GRADIO UI FUNCTIONS
# ============================================================================

def get_system_status():
    """Get current system status"""
    all_images = db.get_all_images()
    total_images = len(all_images)
    total_vectors = vector_db.index.ntotal if vector_db.index else 0
    
    status = f"""
### 📊 System Status
- **Images in Database:** {total_images}
- **Vectors in Index:** {total_vectors}
- **Ollama Connected:** {"✅ Yes" if ollama.is_connected else "❌ No"}
- **LLaVA Model:** {"✅ Available" if Config.VISION_MODEL in ollama.models_available else "❌ Missing"}
- **Embedding Model:** {"✅ Available" if Config.EMBEDDING_MODEL in ollama.models_available else "❌ Missing"}
"""
    
    if total_images == 0:
        status += "\n⚠️ **No images yet - upload some to get started!**"
    elif total_images != total_vectors:
        status += f"\n⚠️ **Warning:** {total_images - total_vectors} images need indexing"
    else:
        status += "\n✅ **All systems ready!**"
    
    return status

def search_interface(query, top_k):
    """Search for images"""
    if not query or len(query) < 3:
        return None, "Please enter at least 3 characters", get_system_status()
    
    results, error = pipeline.search_images(query, int(top_k))
    
    if error:
        return None, f"❌ {error}", get_system_status()
    
    if not results:
        return None, "No results found", get_system_status()
    
    # Format results for display
    gallery_images = []
    result_text = f"✅ Found {len(results)} matching images:\n\n"
    
    for i, result in enumerate(results, 1):
        confidence_pct = result['confidence'] * 100
        if confidence_pct >= 70:
            badge = "🟢 High Match"
        elif confidence_pct >= 40:
            badge = "🟡 Medium Match"
        else:
            badge = "🔴 Low Match"
        
        result_text += f"**#{i} - {result['filename']}**\n"
        result_text += f"{badge} ({confidence_pct:.1f}%)\n"
        result_text += f"📝 {result['description']}\n\n"
        result_text += "---\n\n"
        
        # Add to gallery
        img_path = Path(result['file_path'])
        if img_path.exists():
            gallery_images.append((str(img_path), f"#{i} - Confidence: {confidence_pct:.1f}%"))
    
    return gallery_images, result_text, get_system_status()

def upload_interface(files, progress=gr.Progress()):
    """Upload and process images"""
    if not files:
        return None, "No files selected", get_system_status()
    
    results_text = f"Processing {len(files)} image(s)...\n\n"
    success_count = 0
    gallery_images = []
    
    for i, file in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {i+1}/{len(files)}")
        
        filename = Path(file).name
        results_text += f"**[{i+1}/{len(files)}] {filename}**\n"
        
        success, result = pipeline.process_uploaded_image(file)
        
        if success:
            success_count += 1
            results_text += f"✅ Success!\n"
            results_text += f"📝 Description: {result['description']}\n"
            results_text += f"🆔 Image ID: {result['image_id']}\n\n"
            gallery_images.append((result['file_path'], filename))
        else:
            results_text += f"❌ Failed: {result}\n\n"
    
    results_text += f"\n{'='*50}\n"
    results_text += f"✅ Successfully processed: {success_count}/{len(files)} images\n"
    
    if success_count > 0:
        results_text += "\n🎉 Images are now searchable!"
    
    return gallery_images, results_text, get_system_status()

def rebuild_index():
    """Rebuild FAISS index"""
    count = vector_db.rebuild_from_database(db, ollama)
    if count > 0:
        return f"✅ Successfully rebuilt index with {count} vectors!", get_system_status()
    else:
        return "⚠️ No images to index", get_system_status()

def get_gallery():
    """Get all images for gallery"""
    all_images = db.get_all_images()
    
    if not all_images:
        return None, "📭 No images yet. Upload some images to get started!", get_system_status()
    
    gallery_images = []
    info_text = f"**Total: {len(all_images)} images**\n\n"
    
    for img in all_images:
        img_path = Path(img.file_path)
        if img_path.exists():
            caption = f"{img.filename}\n{img.description[:100]}..."
            gallery_images.append((str(img_path), caption))
            info_text += f"**{img.filename}**\n{img.description}\n\n---\n\n"
    
    return gallery_images, info_text, get_system_status()

def get_image_list():
    """Get list of images for dropdown"""
    all_images = db.get_all_images()
    if not all_images:
        return []
    return [(f"[{img.id}] {img.filename}", img.id) for img in all_images[:50]]  # First 50

def inspect_vector(image_id):
    """Inspect vector for a specific image"""
    if not image_id:
        return "Please select an image", get_system_status()
    
    try:
        img = db.get_image_by_id(int(image_id))
        if not img:
            return "❌ Image not found", get_system_status()
        
        if not img.embedding:
            return "❌ No embedding found for this image", get_system_status()
        
        # Parse embedding
        embedding_data = json.loads(img.embedding)
        vector = np.array(embedding_data, dtype=np.float32)
        
        # Create detailed report
        report = f"""
## 🔍 Vector Details for: {img.filename}

**Description:** {img.description}

### Vector Statistics:
- **Dimension:** {len(vector)}
- **L2 Norm:** {np.linalg.norm(vector):.6f}
- **Mean:** {vector.mean():.6f}
- **Std Dev:** {vector.std():.6f}
- **Min Value:** {vector.min():.6f}
- **Max Value:** {vector.max():.6f}

### First 20 Values:
```
{vector[:20].tolist()}
```

### Vector Visualization:
- First 10 dimensions: {' '.join([f'{v:.3f}' for v in vector[:10]])}
- Middle 10 dimensions: {' '.join([f'{v:.3f}' for v in vector[374:384]])}
- Last 10 dimensions: {' '.join([f'{v:.3f}' for v in vector[-10:]])}

### Health Check:
- ✅ Dimension correct: {len(vector) == Config.EMBEDDING_DIMENSION}
- ✅ Normalized: {abs(np.linalg.norm(vector) - 1.0) < 0.01}
- ✅ No NaN values: {not np.isnan(vector).any()}
- ✅ No Inf values: {not np.isinf(vector).any()}
"""
        
        return report, get_system_status()
        
    except Exception as e:
        return f"❌ Error inspecting vector: {str(e)}", get_system_status()

def compare_vectors(image_id1, image_id2):
    """Compare two image vectors"""
    if not image_id1 or not image_id2:
        return "Please select two images to compare", get_system_status()
    
    try:
        img1 = db.get_image_by_id(int(image_id1))
        img2 = db.get_image_by_id(int(image_id2))
        
        if not img1 or not img2:
            return "❌ One or both images not found", get_system_status()
        
        if not img1.embedding or not img2.embedding:
            return "❌ One or both images missing embeddings", get_system_status()
        
        # Parse embeddings
        vec1 = np.array(json.loads(img1.embedding), dtype=np.float32)
        vec2 = np.array(json.loads(img2.embedding), dtype=np.float32)
        
        # Compute similarity
        cosine_sim = np.dot(vec1, vec2)
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        
        # Element-wise comparison
        diff = vec1 - vec2
        
        report = f"""
## 📊 Vector Comparison

### Image 1: {img1.filename}
**Description:** {img1.description[:100]}...

### Image 2: {img2.filename}
**Description:** {img2.description[:100]}...

### Similarity Metrics:
- **Cosine Similarity:** {cosine_sim:.6f} (1.0 = identical, -1.0 = opposite)
- **Euclidean Distance:** {euclidean_dist:.6f} (0.0 = identical)

### Interpretation:
{'🟢 **Very Similar**' if cosine_sim > 0.8 else '🟡 **Somewhat Similar**' if cosine_sim > 0.5 else '🔴 **Not Similar**'}

### Difference Statistics:
- **Mean Difference:** {diff.mean():.6f}
- **Max Difference:** {diff.max():.6f}
- **Min Difference:** {diff.min():.6f}
- **Std Dev of Difference:** {diff.std():.6f}

### Top 5 Most Different Dimensions:
"""
        
        # Find top 5 most different dimensions
        abs_diff = np.abs(diff)
        top_indices = np.argsort(abs_diff)[-5:][::-1]
        
        for idx in top_indices:
            report += f"\n- Dimension {idx}: {vec1[idx]:.4f} vs {vec2[idx]:.4f} (diff: {diff[idx]:.4f})"
        
        return report, get_system_status()
        
    except Exception as e:
        return f"❌ Error comparing vectors: {str(e)}", get_system_status()

def test_query_vector(query_text):
    """Test query vector generation"""
    if not query_text or len(query_text) < 3:
        return "Please enter at least 3 characters", get_system_status()
    
    try:
        # Generate embedding
        query_embedding = ollama.generate_embedding(query_text)
        
        if query_embedding is None:
            return "❌ Failed to generate embedding", get_system_status()
        
        # Statistics
        report = f"""
## 🔍 Query Vector Analysis

**Query:** "{query_text}"

### Vector Statistics:
- **Dimension:** {len(query_embedding)}
- **L2 Norm:** {np.linalg.norm(query_embedding):.6f}
- **Mean:** {query_embedding.mean():.6f}
- **Std Dev:** {query_embedding.std():.6f}
- **Min Value:** {query_embedding.min():.6f}
- **Max Value:** {query_embedding.max():.6f}

### First 20 Values:
```
{query_embedding[:20].tolist()}
```

### Search Preview (Top 5 matches):
"""
        
        # Do a quick search
        results = vector_db.search(query_embedding, 5)
        
        if results:
            for i, result in enumerate(results, 1):
                img = db.get_image_by_id(result['image_id'])
                if img:
                    report += f"\n**{i}. {img.filename}** (confidence: {result['confidence']:.4f})"
                    report += f"\n   {img.description[:80]}...\n"
        else:
            report += "\nNo matches found."
        
        return report, get_system_status()
        
    except Exception as e:
        return f"❌ Error: {str(e)}", get_system_status()

def export_vectors_csv():
    """Export vectors to CSV"""
    try:
        all_images = db.get_all_images()
        
        if not all_images:
            return "❌ No images to export", get_system_status()
        
        output_file = Config.PROJECT_ROOT / "vectors_export.csv"
        
        with open(output_file, 'w') as f:
            f.write("id,filename,description,dimension,norm,mean,std,min,max\n")
            
            for img in all_images:
                if img.embedding:
                    embedding_data = json.loads(img.embedding)
                    vector = np.array(embedding_data, dtype=np.float32)
                    
                    f.write(f"{img.id},")
                    f.write(f'"{img.filename}",')
                    f.write(f'"{img.description[:50] if img.description else ""}...",')
                    f.write(f"{len(vector)},")
                    f.write(f"{np.linalg.norm(vector):.6f},")
                    f.write(f"{vector.mean():.6f},")
                    f.write(f"{vector.std():.6f},")
                    f.write(f"{vector.min():.6f},")
                    f.write(f"{vector.max():.6f}\n")
        
        return f"✅ Exported {len(all_images)} vectors to: {output_file}", get_system_status()
        
    except Exception as e:
        return f"❌ Export failed: {str(e)}", get_system_status()

# ============================================================================
# CREATE GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="🔍 AI Image Search", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🔍 AI Image Search System
    ### Powered by LLaVA Vision AI & FAISS Vector Search
    Search your images using natural language descriptions!
    """)
    
    # Status sidebar
    with gr.Row():
        with gr.Column(scale=3):
            pass  # Main content will go here
        with gr.Column(scale=1):
            status_box = gr.Markdown(get_system_status())
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            rebuild_btn = gr.Button("🔨 Rebuild Index", size="sm")
            rebuild_output = gr.Textbox(label="Rebuild Status", visible=False)
    
    # Main tabs
    with gr.Tabs():
        
        # SEARCH TAB
        with gr.TabItem("🔍 Search"):
            gr.Markdown("### Search for images using natural language")
            
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., sunset over mountains, person with a dog, red car...",
                    scale=4
                )
                top_k = gr.Dropdown(
                    choices=[5, 10, 15, 20],
                    value=5,
                    label="Results",
                    scale=1
                )
            
            search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
            
            with gr.Row():
                search_results_text = gr.Markdown()
            
            search_gallery = gr.Gallery(
                label="Search Results",
                columns=3,
                height="auto"
            )
        
        # UPLOAD TAB
        with gr.TabItem("📤 Upload"):
            gr.Markdown("### Upload images to make them searchable")
            
            upload_files = gr.File(
                label="Select Images",
                file_count="multiple",
                file_types=["image"]
            )
            
            upload_btn = gr.Button("🚀 Process Images", variant="primary", size="lg")
            
            upload_results_text = gr.Markdown()
            upload_gallery = gr.Gallery(
                label="Uploaded Images",
                columns=3,
                height="auto"
            )
        
        # GALLERY TAB
        with gr.TabItem("📋 Gallery"):
            gr.Markdown("### Browse all your images")
            
            gallery_refresh_btn = gr.Button("🔄 Refresh Gallery")
            
            gallery_info = gr.Markdown()
            gallery_display = gr.Gallery(
                label="All Images",
                columns=4,
                height="auto"
            )
        
        # VECTOR INSPECTOR TAB
        with gr.TabItem("🔬 Vector Inspector"):
            gr.Markdown("### Inspect and analyze image embeddings")
            
            with gr.Tabs():
                # Sub-tab 1: Inspect Single Vector
                with gr.TabItem("🔍 Inspect Vector"):
                    gr.Markdown("View detailed vector information for any image")
                    
                    with gr.Row():
                        image_dropdown = gr.Dropdown(
                            label="Select Image",
                            choices=get_image_list(),
                            value=get_image_list()[0][1] if get_image_list() else None
                        )
                        inspect_btn = gr.Button("🔍 Inspect", variant="primary")
                    
                    inspect_output = gr.Markdown()
                
                # Sub-tab 2: Compare Vectors
                with gr.TabItem("📊 Compare Vectors"):
                    gr.Markdown("Compare similarity between two image vectors")
                    
                    with gr.Row():
                        image_dropdown1 = gr.Dropdown(
                            label="Image 1",
                            choices=get_image_list(),
                            value=get_image_list()[0][1] if get_image_list() else None
                        )
                        image_dropdown2 = gr.Dropdown(
                            label="Image 2",
                            choices=get_image_list(),
                            value=get_image_list()[1][1] if len(get_image_list()) > 1 else None
                        )
                    
                    compare_btn = gr.Button("📊 Compare", variant="primary")
                    compare_output = gr.Markdown()
                
                # Sub-tab 3: Test Query Vector
                with gr.TabItem("🧪 Test Query"):
                    gr.Markdown("Generate and analyze embedding for a text query")
                    
                    query_input = gr.Textbox(
                        label="Enter query text",
                        placeholder="e.g., Nike shoes, person standing, motorcycle..."
                    )
                    query_test_btn = gr.Button("🧪 Test Query", variant="primary")
                    query_output = gr.Markdown()
                
                # Sub-tab 4: Export Vectors
                with gr.TabItem("💾 Export"):
                    gr.Markdown("Export all vectors to CSV for external analysis")
                    
                    export_btn = gr.Button("💾 Export to CSV", variant="primary", size="lg")
                    export_output = gr.Markdown()
            
            vector_status = gr.Markdown()
    
    # Event handlers
    search_btn.click(
        fn=search_interface,
        inputs=[search_query, top_k],
        outputs=[search_gallery, search_results_text, status_box]
    )
    
    upload_btn.click(
        fn=upload_interface,
        inputs=[upload_files],
        outputs=[upload_gallery, upload_results_text, status_box]
    )
    
    gallery_refresh_btn.click(
        fn=get_gallery,
        outputs=[gallery_display, gallery_info, status_box]
    )
    
    refresh_btn.click(
        fn=lambda: get_system_status(),
        outputs=[status_box]
    )
    
    rebuild_btn.click(
        fn=rebuild_index,
        outputs=[rebuild_output, status_box]
    )
    
    # Vector Inspector event handlers
    inspect_btn.click(
        fn=inspect_vector,
        inputs=[image_dropdown],
        outputs=[inspect_output, vector_status]
    )
    
    compare_btn.click(
        fn=compare_vectors,
        inputs=[image_dropdown1, image_dropdown2],
        outputs=[compare_output, vector_status]
    )
    
    query_test_btn.click(
        fn=test_query_vector,
        inputs=[query_input],
        outputs=[query_output, vector_status]
    )
    
    export_btn.click(
        fn=export_vectors_csv,
        outputs=[export_output, vector_status]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting AI Image Search System...")
    print(f"📁 Images folder: {Config.IMAGES_FOLDER}")
    print(f"💾 Database: {Config.DATABASE_PATH}")
    print(f"🔢 Vector index: {Config.FAISS_INDEX_PATH}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

