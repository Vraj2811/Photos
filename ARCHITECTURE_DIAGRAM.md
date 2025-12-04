# Image Search System - Architecture Diagram

## System Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI1[Streamlit UI]
        UI2[Gradio UI]
        UI3[React Frontend]
    end
    
    subgraph "Application Layer"
        APP[Main Application]
        API[FastAPI Backend]
        PIPELINE[Image Pipeline]
    end
    
    subgraph "AI Processing Layer"
        OLLAMA[Ollama Service]
        LLAVA[LLaVA Vision Model]
        EMBED[Nomic Embed Text Model]
    end
    
    subgraph "Storage Layer"
        DB[(SQLite Database)]
        FAISS[FAISS Vector Index]
        FILES[Image Files]
    end
    
    UI1 --> APP
    UI2 --> APP
    UI3 --> API
    
    APP --> PIPELINE
    API --> PIPELINE
    
    PIPELINE --> OLLAMA
    OLLAMA --> LLAVA
    OLLAMA --> EMBED
    
    PIPELINE --> DB
    PIPELINE --> FAISS
    PIPELINE --> FILES
    
    style OLLAMA fill:#667eea
    style DB fill:#4CAF50
    style FAISS fill:#FF9800
    style LLAVA fill:#E91E63
    style EMBED fill:#9C27B0
```

## Detailed Data Flow

### Upload & Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Frontend UI
    participant Pipeline as Image Pipeline
    participant Ollama as Ollama Service
    participant LLaVA as LLaVA Model
    participant Embedder as Embedding Model
    participant DB as SQLite Database
    participant FAISS as FAISS Index
    participant Storage as File Storage

    User->>UI: Upload Image(s)
    UI->>Pipeline: Process Images
    
    Pipeline->>Storage: Save Image File
    Storage-->>Pipeline: File Path
    
    Pipeline->>DB: Store Image Metadata
    DB-->>Pipeline: Image ID
    
    Pipeline->>Ollama: Generate Description
    Ollama->>LLaVA: Analyze Image
    LLaVA-->>Ollama: Image Description
    Ollama-->>Pipeline: Description Text
    
    Pipeline->>DB: Update with Description
    
    Pipeline->>Ollama: Generate Embedding
    Ollama->>Embedder: Embed Description
    Embedder-->>Ollama: 768-dim Vector
    Ollama-->>Pipeline: Embedding Vector
    
    Pipeline->>FAISS: Add Vector to Index
    FAISS-->>Pipeline: Index Updated
    
    Pipeline->>DB: Update Processing Status
    Pipeline-->>UI: Processing Complete
    UI-->>User: Show Results
```

### Search Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Frontend UI
    participant Pipeline as Image Pipeline
    participant Ollama as Ollama Service
    participant Embedder as Embedding Model
    participant FAISS as FAISS Index
    participant DB as SQLite Database

    User->>UI: Enter Search Query
    UI->>Pipeline: Search Request
    
    Pipeline->>Ollama: Generate Query Embedding
    Ollama->>Embedder: Embed Query Text
    Embedder-->>Ollama: Query Vector
    Ollama-->>Pipeline: Query Vector
    
    Pipeline->>FAISS: Vector Similarity Search
    FAISS-->>Pipeline: Top K Similar Vectors (IDs + Scores)
    
    Pipeline->>DB: Fetch Image Metadata
    DB-->>Pipeline: Image Details
    
    Pipeline-->>UI: Ranked Results
    UI-->>User: Display Images with Scores
```

## Component Architecture

```mermaid
graph LR
    subgraph "Core Components"
        A[ImageDB] --> B[Database Operations]
        C[OllamaProcessor] --> D[AI Processing]
        E[VectorDB] --> F[FAISS Operations]
        G[ImagePipeline] --> H[Orchestration]
    end
    
    H --> A
    H --> C
    H --> E
    
    style A fill:#4CAF50
    style C fill:#667eea
    style E fill:#FF9800
    style G fill:#E91E63
```

## Database Schema

```mermaid
erDiagram
    IMAGES {
        int id PK
        string filename
        string file_path
        text description
        datetime created_at
        datetime processed_at
    }
    
    FAISS_INDEX {
        int vector_id
        int image_id FK
        float[] embedding
    }
    
    ID_MAPPING {
        int faiss_id
        int database_id
    }
    
    IMAGES ||--o{ FAISS_INDEX : has
    IMAGES ||--o{ ID_MAPPING : maps
```

## System States

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing : Upload Images
    Processing --> GeneratingDescription : Image Saved
    GeneratingDescription --> CreatingEmbedding : Description Complete
    CreatingEmbedding --> IndexingVector : Embedding Generated
    IndexingVector --> Complete : Vector Indexed
    Complete --> Idle : Ready for Search
    
    Idle --> Searching : User Query
    Searching --> RetrievingResults : Vector Search
    RetrievingResults --> DisplayingResults : Fetch Metadata
    DisplayingResults --> Idle : Results Shown
```

## Technology Stack

```mermaid
graph TB
    subgraph "Frontend Technologies"
        ST[Streamlit 1.28+]
        GR[Gradio 4.0+]
        RE[React + TypeScript]
    end
    
    subgraph "Backend Technologies"
        FA[FastAPI]
        SA[SQLAlchemy 2.0+]
        SQ[SQLite]
    end
    
    subgraph "AI/ML Stack"
        OL[Ollama]
        LV[LLaVA Vision Model]
        NE[Nomic Embed Text]
        FS[FAISS CPU 1.7.4+]
    end
    
    subgraph "Utilities"
        PIL[Pillow 10.0+]
        NP[NumPy 1.24+]
        PD[Pandas 2.0+]
    end
    
    ST --> FA
    GR --> FA
    RE --> FA
    
    FA --> SA
    SA --> SQ
    
    FA --> OL
    OL --> LV
    OL --> NE
    
    FA --> FS
    
    style OL fill:#667eea
    style FS fill:#FF9800
    style LV fill:#E91E63
```

## Deployment Options

```mermaid
graph TB
    subgraph "Development"
        D1[Local Python]
        D2[Ollama Local]
        D3[SQLite]
    end
    
    subgraph "Production - Docker"
        P1[Docker Compose]
        P2[Frontend Container]
        P3[Backend Container]
        P4[Ollama Container]
        P5[Volume: Database]
        P6[Volume: Images]
    end
    
    subgraph "Cloud Deployment"
        C1[Cloud VM]
        C2[Container Registry]
        C3[Cloud Storage]
        C4[Load Balancer]
    end
    
    D1 --> P1
    P1 --> P2
    P1 --> P3
    P1 --> P4
    P1 --> P5
    P1 --> P6
    
    P1 --> C1
    P2 --> C2
    C1 --> C3
    C1 --> C4
```

## Performance Optimization Flow

```mermaid
graph TD
    A[Image Upload] --> B{Batch Size Check}
    B -->|< 10 images| C[Process Sequentially]
    B -->|> 10 images| D[Batch Processing]
    
    C --> E[Generate Description]
    D --> F[Parallel Description Gen]
    
    E --> G[Generate Embedding]
    F --> H[Batch Embeddings]
    
    G --> I[Add to FAISS]
    H --> J[Bulk FAISS Insert]
    
    I --> K[Update Database]
    J --> L[Batch DB Update]
    
    K --> M[Complete]
    L --> M
    
    style D fill:#4CAF50
    style F fill:#4CAF50
    style H fill:#4CAF50
    style J fill:#4CAF50
```

## Error Handling & Recovery

```mermaid
graph TB
    START[Start Processing] --> CHECK{Ollama Running?}
    
    CHECK -->|No| ERROR1[Connection Error]
    CHECK -->|Yes| MODELS{Models Available?}
    
    MODELS -->|No| ERROR2[Model Missing Error]
    MODELS -->|Yes| PROCESS[Process Image]
    
    PROCESS --> DESC{Description Generated?}
    DESC -->|No| RETRY1[Retry 3x]
    DESC -->|Yes| EMBED{Embedding Generated?}
    
    RETRY1 -->|Failed| ERROR3[Skip Image + Log]
    RETRY1 -->|Success| EMBED
    
    EMBED -->|No| RETRY2[Retry 3x]
    EMBED -->|Yes| SAVE[Save to DB & Index]
    
    RETRY2 -->|Failed| ERROR4[Skip Image + Log]
    RETRY2 -->|Success| SAVE
    
    SAVE --> SUCCESS[Complete]
    
    ERROR1 --> NOTIFY[Notify User]
    ERROR2 --> NOTIFY
    ERROR3 --> NOTIFY
    ERROR4 --> NOTIFY
    
    style SUCCESS fill:#4CAF50
    style ERROR1 fill:#f44336
    style ERROR2 fill:#f44336
    style ERROR3 fill:#FF9800
    style ERROR4 fill:#FF9800
```

---

## Quick Reference

### Key Metrics
- **Embedding Dimension**: 768
- **Default Search Results**: 10
- **Vector Index Type**: FAISS Flat (L2)
- **Database**: SQLite with SQLAlchemy ORM

### API Endpoints (if using FastAPI backend)
```
POST   /api/upload          - Upload images
POST   /api/search          - Search images
GET    /api/images          - List all images
GET    /api/images/{id}     - Get image details
DELETE /api/images/{id}     - Delete image
POST   /api/rebuild-index   - Rebuild FAISS index
GET    /api/status          - System status
```

### File Structure
```
Image_Search/
├── streamlit_app.py      # Streamlit UI
├── gradio_app.py         # Gradio UI  
├── backend/
│   └── api.py           # FastAPI backend
├── frontend/            # React frontend
├── images/              # Uploaded images
├── faiss_indexes/       # Vector indexes
└── image_search.db      # SQLite database
```

