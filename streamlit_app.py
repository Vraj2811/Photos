import streamlit as st
from pathlib import Path
import json
import time
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import traceback

# Import required libraries
try:
    import ollama
    import faiss
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as e:
    st.error(f"❌ Missing required packages: {e}")
    st.info("💡 Install with: pip install streamlit ollama-python faiss-cpu pillow numpy pandas sqlalchemy")
    st.stop()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    PROJECT_ROOT = Path(__file__).parent
    DATABASE_PATH = PROJECT_ROOT / "image_search.db"
    IMAGES_FOLDER = PROJECT_ROOT / "images"
    FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"
    
    # Model configuration
    VISION_MODEL = "llava"
    EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_DIMENSION = 768
    
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Ensure directories exist
Config.IMAGES_FOLDER.mkdir(exist_ok=True)
Config.FAISS_INDEX_PATH.mkdir(exist_ok=True)

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="🔍 AI Image Search",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Styling
st.markdown("""
<style>
    /* Force white background */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Image cards */
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #e0e0e0;
    }
    
    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    
    .confidence-high {
        background: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
        color: white;
    }
    
    /* Search box */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Upload area */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        background-color: #f0f2f6 !important;
        color: #1f1f1f !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    [data-testid="stSidebarNav"] {
        background-color: #f8f9fa !important;
    }
    
    /* Ensure readable text colors on white background */
    h1, h2, h3, h4, h5, h6 {
        color: #1f1f1f !important;
    }
    
    p, div, span, label {
        color: #262730 !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #262730 !important;
    }
    
    /* Success/Info/Warning boxes should keep their colors */
    [data-testid="stSuccess"],
    [data-testid="stInfo"],
    [data-testid="stWarning"],
    [data-testid="stError"],
    .stSuccess, .stInfo, .stWarning, .stError {
        color: inherit !important;
    }
    
    [data-testid="stSuccess"] *,
    [data-testid="stInfo"] *,
    [data-testid="stWarning"] *,
    [data-testid="stError"] * {
        color: inherit !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        color: #1f1f1f !important;
    }
    
    /* Button text should be dark */
    .stButton > button {
        color: #1f1f1f !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE SETUP
# ============================================================================

Base = declarative_base()

class ImageRecord(Base):
    """Database model for storing image metadata"""
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    description = Column(Text)
    embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

@st.cache_resource
def init_database():
    """Initialize database and return session factory"""
    try:
        engine = create_engine(Config.DATABASE_URL, echo=False)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        return None

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class ImageDB:
    """Database operations for image records"""
    
    def __init__(self, session_factory):
        self.SessionLocal = session_factory
    
    def add_image(self, filename, file_path, description, embedding):
        """Add new image to database"""
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
        """Get all image records"""
        session = self.SessionLocal()
        try:
            return session.query(ImageRecord).order_by(ImageRecord.created_at.desc()).all()
        finally:
            session.close()
    
    def get_image_by_id(self, image_id):
        """Get specific image by ID"""
        session = self.SessionLocal()
        try:
            return session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        finally:
            session.close()

# ============================================================================
# OLLAMA INTEGRATION
# ============================================================================

class OllamaProcessor:
    """Handle Ollama model interactions"""
    
    def __init__(self):
        self.vision_model = Config.VISION_MODEL
        self.embedding_model = Config.EMBEDDING_MODEL
        self.models_available = []
        self.is_connected = self.check_models()
    
    def check_models(self):
        """Check if required models are available"""
        try:
            models_response = ollama.list()
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
        """Generate 2-3 line description using LLaVA"""
        try:
            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in 2-3 clear, concise sentences. Focus on the main subject, key details, and overall scene.',
                    'images': [str(image_path)]
                }]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Image description failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_embedding(self, text):
        """Convert text to embedding vector"""
        try:
            if not text or not text.strip():
                print("Warning: Empty text provided for embedding")
                return None
                
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            
            if 'embedding' in response and response['embedding']:
                embedding = np.array(response['embedding'], dtype=np.float32)
                
                # Ensure correct dimension
                if len(embedding) > Config.EMBEDDING_DIMENSION:
                    embedding = embedding[:Config.EMBEDDING_DIMENSION]
                elif len(embedding) < Config.EMBEDDING_DIMENSION:
                    padded = np.zeros(Config.EMBEDDING_DIMENSION, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded
                
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
            else:
                print(f"Embedding response missing 'embedding' key: {response}")
                return None
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            traceback.print_exc()
            return None

# ============================================================================
# FAISS VECTOR DATABASE
# ============================================================================

class VectorDB:
    """FAISS vector database for similarity search"""
    
    def __init__(self):
        self.index_file = Config.FAISS_INDEX_PATH / "vectors.index"
        self.mapping_file = Config.FAISS_INDEX_PATH / "mapping.json"
        self.index = None
        self.id_mapping = {}
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing index or create new one"""
        if self.index_file.exists() and self.mapping_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    self.id_mapping = {int(k): v for k, v in mapping_data.items()}
                print(f"✓ Loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Failed to load index: {e}")
                self.create_new_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatIP(Config.EMBEDDING_DIMENSION)
        self.id_mapping = {}
    
    def add_vector(self, embedding, image_id):
        """Add vector to index"""
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            self.index.add(embedding)
            faiss_idx = self.index.ntotal - 1
            self.id_mapping[faiss_idx] = image_id
            self.save_index()
            return True
        except Exception as e:
            print(f"Failed to add vector: {e}")
            traceback.print_exc()
            return False
    
    def search(self, query_embedding, top_k=5):
        """Search for similar vectors"""
        try:
            if self.index.ntotal == 0:
                print("Index is empty - no vectors to search")
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
            
            print(f"Search found {len(results)} results")
            return results
        except Exception as e:
            print(f"Search failed: {e}")
            traceback.print_exc()
            return []
    
    def save_index(self):
        """Save index to disk"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in self.id_mapping.items()}
                json.dump(mapping_str, f, indent=2)
        except Exception as e:
            print(f"Failed to save index: {e}")
    
    def rebuild_from_database(self, db, ollama_processor):
        """Rebuild entire index from database records"""
        try:
            all_images = db.get_all_images()
            if not all_images:
                return 0
            
            # Create fresh index
            self.create_new_index()
            
            count = 0
            for img in all_images:
                embedding = None
                
                # Try to load existing embedding
                if img.embedding:
                    try:
                        embedding_data = json.loads(img.embedding)
                        embedding = np.array(embedding_data, dtype=np.float32)
                        
                        # Normalize
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                    except Exception as e:
                        print(f"Failed to load embedding for image {img.id}: {e}")
                
                # Generate embedding if missing
                if embedding is None and img.description:
                    embedding = ollama_processor.generate_embedding(img.description)
                
                if embedding is not None:
                    if self.add_vector(embedding, img.id):
                        count += 1
            
            return count
        except Exception as e:
            print(f"Rebuild failed: {e}")
            traceback.print_exc()
            return 0

# ============================================================================
# IMAGE PROCESSING PIPELINE
# ============================================================================

class ImagePipeline:
    """Main pipeline for processing and searching images"""
    
    def __init__(self, db, ollama_processor, vector_db):
        self.db = db
        self.ollama = ollama_processor
        self.vector_db = vector_db
    
    def process_uploaded_image(self, uploaded_file):
        """Process a newly uploaded image"""
        try:
            # Generate unique filename
            timestamp = int(time.time())
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
            filename = f"{timestamp}_{file_hash}_{uploaded_file.name}"
            file_path = Config.IMAGES_FOLDER / filename
            
            # Save image
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Generate description using LLaVA
            description = self.ollama.generate_description(file_path)
            if not description:
                return False, "Failed to generate description - Check if Ollama and LLaVA are running"
            
            # Generate embedding from description
            embedding = self.ollama.generate_embedding(description)
            if embedding is None:
                return False, "Failed to generate embedding - Check if embedding model is available"
            
            # Store in database
            image_id = self.db.add_image(filename, str(file_path), description, embedding)
            if not image_id:
                return False, "Failed to save to database"
            
            # Add to vector database
            success = self.vector_db.add_vector(embedding, image_id)
            if not success:
                return False, "Failed to add to vector index"
            
            return True, {
                'filename': filename,
                'description': description,
                'image_id': image_id,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            print(f"Processing error: {e}")
            traceback.print_exc()
            return False, f"Processing error: {str(e)}"
    
    def search_images(self, query, top_k=5):
        """Search for images similar to query"""
        try:
            print(f"Searching for: '{query}' (top_k={top_k})")
            
            # Check if index has vectors
            if self.vector_db.index.ntotal == 0:
                return [], "No images in the database yet. Please upload some images first."
            
            # Convert query to embedding
            query_embedding = self.ollama.generate_embedding(query)
            if query_embedding is None:
                return [], "Failed to generate embedding for query. Check if Ollama is running."
            
            # Search in vector database
            vector_results = self.vector_db.search(query_embedding, top_k)
            
            if not vector_results:
                return [], "No matching results found. Try a different search query."
            
            # Get full image records
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
            
            # Sort by confidence (descending)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results, None
            
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return [], error_msg

# ============================================================================
# INITIALIZE SYSTEM
# ============================================================================

@st.cache_resource
def init_system():
    """Initialize all system components"""
    SessionLocal = init_database()
    if not SessionLocal:
        st.error("Failed to initialize database")
        st.stop()
    
    db = ImageDB(SessionLocal)
    ollama_processor = OllamaProcessor()
    vector_db = VectorDB()
    pipeline = ImagePipeline(db, ollama_processor, vector_db)
    
    return pipeline, db, ollama_processor, vector_db

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_confidence_badge(confidence):
    """Get HTML badge for confidence score"""
    if confidence >= 0.7:
        badge_class = "confidence-high"
        label = "High Match"
    elif confidence >= 0.4:
        badge_class = "confidence-medium"
        label = "Medium Match"
    else:
        badge_class = "confidence-low"
        label = "Low Match"
    
    return f'<span class="confidence-badge {badge_class}">{label}: {confidence:.2%}</span>'

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Image Search System</h1>
        <p>Powered by LLaVA Vision AI & FAISS Vector Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    pipeline, db, ollama, vector_db = init_system()
    
    # Sidebar - System Status
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        all_images = db.get_all_images()
        total_images = len(all_images)
        total_vectors = vector_db.index.ntotal if vector_db.index else 0
        
        # Modern stat cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_images}</div>
                <div class="stat-label">Images</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_vectors}</div>
                <div class="stat-label">Vectors</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if total_images == 0:
            st.info("👆 Upload images to get started!")
        elif total_images == total_vectors and total_images > 0:
            st.success("✅ All systems ready!")
        elif total_images > total_vectors:
            st.warning(f"⚠️ {total_images - total_vectors} images need indexing")
            if st.button("🔨 Rebuild Index Now", use_container_width=True):
                with st.spinner("Rebuilding..."):
                    count = vector_db.rebuild_from_database(db, ollama)
                    st.success(f"✅ Rebuilt {count} vectors!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        
        # System Actions
        st.markdown("### 🛠️ Quick Actions")
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        if st.button("🔨 Rebuild Vector Index", use_container_width=True):
            with st.spinner("Rebuilding index..."):
                count = vector_db.rebuild_from_database(db, ollama)
                if count > 0:
                    st.success(f"✅ Rebuilt {count} vectors!")
                else:
                    st.warning("No images to index")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
        # Ollama Status
        st.markdown("### 🤖 AI Models")
        
        if ollama.is_connected:
            # Check Vision Model
            if Config.VISION_MODEL in ollama.models_available:
                st.success(f"✅ {Config.VISION_MODEL}")
            else:
                st.error(f"❌ {Config.VISION_MODEL}")
                st.caption("`ollama pull llava`")
            
            # Check Embedding Model
            if Config.EMBEDDING_MODEL in ollama.models_available:
                st.success(f"✅ {Config.EMBEDDING_MODEL}")
            else:
                st.error(f"❌ {Config.EMBEDDING_MODEL}")
                st.caption("`ollama pull nomic-embed-text`")
        else:
            st.error("❌ Ollama not connected")
            st.caption("Start with: `ollama serve`")
        
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"**DB Path:** {Config.DATABASE_PATH}")
            st.write(f"**Images Folder:** {Config.IMAGES_FOLDER}")
            st.write(f"**Index File Exists:** {vector_db.index_file.exists()}")
            st.write(f"**Mapping File Exists:** {vector_db.mapping_file.exists()}")
            st.write(f"**FAISS Index Total:** {vector_db.index.ntotal if vector_db.index else 0}")
    
    # Main Content - Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload", "📋 Gallery"])
    
    # ========================================================================
    # TAB 1: SEARCH
    # ========================================================================
    with tab1:
        st.markdown("### 🔍 Search Images by Description")
        st.markdown("Enter a description of what you're looking for:")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., sunset over mountains, person with a dog, red car...",
                label_visibility="collapsed"
            )
        with col2:
            top_k = st.selectbox("Top", [5, 10, 15, 20], index=0, label_visibility="collapsed")
        
        # Auto-search when query is entered
        if query and len(query) > 2:
            with st.spinner(f"🔍 Searching for '{query}'..."):
                results, error = pipeline.search_images(query, top_k)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 **Troubleshooting:**\n- Make sure Ollama is running: `ollama serve`\n- Check if models are installed\n- Upload images first if database is empty")
            elif results:
                st.success(f"✅ Found {len(results)} matching images")
                st.markdown("---")
                
                # Display results in cards
                for i, result in enumerate(results, 1):
                    # Create card
                    with st.container():
                        col_img, col_info = st.columns([1, 2])
                        
                        with col_img:
                            img_path = Path(result['file_path'])
                            if img_path.exists():
                                st.image(str(img_path), use_container_width=True)
                            else:
                                st.warning("⚠️ Image file not found")
                        
                        with col_info:
                            st.markdown(f"### #{i} {result['filename']}")
                            st.markdown(get_confidence_badge(result['confidence']), unsafe_allow_html=True)
                            st.markdown("**Description:**")
                            st.write(result['description'])
                            
                            # Additional details in expander
                            with st.expander("📊 More Details"):
                                st.write(f"**Image ID:** {result['image_id']}")
                                st.write(f"**Confidence Score:** {result['confidence']:.6f}")
                                st.write(f"**File Path:** {result['file_path']}")
                        
                        st.markdown("---")
            else:
                st.info("🔍 No results found. Try a different search term or upload more images.")
        elif query:
            st.info("⌨️ Keep typing... (minimum 3 characters)")
        else:
            # Show example searches
            st.info("💡 **Try searching for:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("- 🌅 sunset\n- 🏔️ mountains\n- 🌊 ocean")
            with col2:
                st.markdown("- 👤 person\n- 🐕 dog\n- 🚗 car")
            with col3:
                st.markdown("- 🍕 food\n- 💻 technology\n- 🏙️ city")
    
    # ========================================================================
    # TAB 2: UPLOAD
    # ========================================================================
    with tab2:
        st.markdown("### 📤 Upload New Images")
        st.markdown("Select images and we'll automatically generate AI descriptions and index them for search.")
        
        uploaded_files = st.file_uploader(
            "Choose images",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_container = st.container()
                
                success_count = 0
                results_container = st.container()
                
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    with status_container:
                        st.text(f"⏳ Processing {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    success, result = pipeline.process_uploaded_image(file)
                    
                    with results_container:
                        if success:
                            success_count += 1
                            with st.expander(f"✅ {file.name}", expanded=False):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    img = Image.open(file)
                                    st.image(img, use_container_width=True)
                                with col2:
                                    st.write(f"**Saved as:** `{result['filename']}`")
                                    st.write(f"**Image ID:** {result['image_id']}")
                                    st.write(f"**Description:**")
                                    st.info(result['description'])
                        else:
                            st.error(f"❌ {file.name}: {result}")
                
                status_container.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.balloons()
                    st.success(f"🎉 Successfully processed {success_count} of {len(uploaded_files)} images!")
                    st.info("💡 Go to the Search tab to find your images!")
                else:
                    st.error("❌ Failed to process any images. Check Ollama connection.")
        else:
            # Upload preview
            st.info("👆 Click above to select image files")
            st.markdown("""
            **Supported formats:** PNG, JPG, JPEG, WebP, BMP
            
            **What happens when you upload:**
            1. 📸 Image is saved to the database
            2. 🤖 LLaVA AI generates a description
            3. 🔢 Description is converted to vector
            4. 💾 Everything is stored and indexed
            5. 🔍 Ready to search!
            """)
    
    # ========================================================================
    # TAB 3: GALLERY
    # ========================================================================
    with tab3:
        st.markdown("### 📋 Image Gallery")
        
        all_images = db.get_all_images()
        
        if all_images:
            st.markdown(f"**Total: {len(all_images)} images**")
            
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                search_filter = st.text_input("🔍 Filter by description", placeholder="Type to filter...")
            with col2:
                sort_order = st.selectbox("Sort", ["Newest", "Oldest"])
            
            # Apply filter
            if search_filter:
                filtered_images = [img for img in all_images if search_filter.lower() in (img.description or "").lower()]
            else:
                filtered_images = all_images
            
            # Apply sort
            if sort_order == "Oldest":
                filtered_images = list(reversed(filtered_images))
            
            st.markdown(f"*Showing {len(filtered_images)} images*")
            st.markdown("---")
            
            # Display in grid (4 columns)
            cols = st.columns(4)
            for i, img in enumerate(filtered_images):
                with cols[i % 4]:
                    img_path = Path(img.file_path)
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                        with st.expander(f"📄 {img.filename[:20]}..."):
                            st.write(f"**ID:** {img.id}")
                            st.write(f"**Created:** {img.created_at.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Description:**")
                            st.caption(img.description or "No description")
                    else:
                        st.error("❌ File not found")
                        st.caption(img.filename)
        else:
            st.info("📭 No images yet!")
            st.markdown("""
            ### Get Started:
            1. Go to the **Upload** tab
            2. Select some images
            3. Click **Process All Images**
            4. Come back here to see your gallery!
            """)

if __name__ == "__main__":
    main()
