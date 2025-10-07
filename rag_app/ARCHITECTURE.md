# RAG Application - Architecture & Implementation Guide

## 🎯 Overview

A production-ready RAG (Retrieval-Augmented Generation) application that enables Q&A over GitHub repositories using semantic search and LLM-powered responses.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend                        │
│  - Repository ingestion UI                                      │
│  - Multi-repository selector                                    │
│  - Chat interface with source citations                         │
│  - Session management                                           │
└────────────────────────────┬───────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Repository Router│  │  Chat Router │  │  Health Router   │ │
│  │ - Ingest         │  │ - Search     │  │ - Health check   │ │
│  │ - List           │  │ - Chat       │  │                  │ │
│  │ - Delete         │  │ - History    │  │                  │ │
│  └────────┬─────────┘  └──────┬───────┘  └──────────────────┘ │
│           │                   │                                 │
│  ┌────────▼──────────┐  ┌────▼──────────┐                     │
│  │ Ingestion Service │  │  RAG Service  │                     │
│  │ - Load repo       │  │ - Search      │                     │
│  │ - Chunk docs      │  │ - Generate    │                     │
│  │ - Index vectors   │  │ - Memory mgmt │                     │
│  └────────┬──────────┘  └──────┬────────┘                     │
└───────────┼────────────────────┼──────────────────────────────┘
            │                    │
    ┌───────▼────────┐  ┌────────▼────────┐
    │   PostgreSQL   │  │    ChromaDB     │
    │                │  │                 │
    │ - Repositories │  │ - Embeddings    │
    │ - Metadata     │  │ - Vector search │
    └────────────────┘  └─────────────────┘
```

## 📁 Project Structure

```
rag_app/
├── src/
│   ├── backend/
│   │   ├── routers/
│   │   │   ├── repository.py    # Repo ingestion endpoints
│   │   │   ├── chat.py          # Chat & search endpoints
│   │   │   └── health.py        # Health check
│   │   ├── services/
│   │   │   ├── ingestion.py     # Ingestion pipeline
│   │   │   └── rag.py           # RAG service
│   │   ├── schemas/
│   │   │   ├── repository.py    # Repo schemas
│   │   │   └── chat.py          # Chat schemas
│   │   └── app.py               # FastAPI app
│   ├── frontend/
│   │   └── app.py               # Streamlit UI
│   ├── core/
│   │   ├── ingestion.py         # Data loader
│   │   ├── chunking.py          # Chunking strategies
│   │   ├── llm.py               # LLM client
│   │   └── memory.py            # Conversation memory
│   ├── db/
│   │   ├── chroma/
│   │   │   └── client.py        # ChromaDB client
│   │   ├── models.py            # SQLAlchemy models
│   │   └── database.py          # DB session
│   ├── config/
│   │   └── settings.py          # Configuration
│   └── logger.py                # Logging
├── Dockerfile.backend           # Backend container
├── Dockerfile.frontend          # Frontend container
├── docker-compose.yml           # Orchestration
├── pyproject.toml              # Dependencies
├── .env.example                # Env template
└── README_SETUP.md             # Setup guide
```

## 🔄 Data Flow

### 1. Repository Ingestion Flow

```
User enters GitHub URL
        ↓
Frontend → POST /repository/ingest
        ↓
Backend validates URL
        ↓
Check PostgreSQL for existing repo
        ↓
Download .md/.mdx files from GitHub
        ↓
Apply chunking strategy (section/paragraph/sliding_window/llm)
        ↓
Generate embeddings (sentence-transformers)
        ↓
Store in ChromaDB with metadata
        ↓
Update PostgreSQL with repo metadata
        ↓
Return success to frontend
```

### 2. Chat/Q&A Flow

```
User asks question
        ↓
Frontend → POST /chat/chat
        ↓
RAG Service performs semantic search
        ↓
ChromaDB returns top-k similar chunks
        ↓
Build context from search results
        ↓
Retrieve conversation history from memory
        ↓
Construct prompt with context
        ↓
OpenAI generates response
        ↓
Update conversation memory
        ↓
Return answer + sources to frontend
```

## 🧩 Core Components

### 1. Data Loader ([ingestion.py](rag_app/src/core/ingestion.py))

**Purpose**: Download and extract markdown files from GitHub repositories

**Key Features**:
- GitHub URL validation
- Download entire repo as ZIP
- Extract `.md` and `.mdx` files
- Parse frontmatter metadata
- Error handling

**Methods**:
- `load_repo_from_url()` - Main entry point
- `_validate_url()` - Ensures valid GitHub URL
- `_parse_url()` - Extracts owner/repo name

### 2. Chunking Strategies ([chunking.py](rag_app/src/core/chunking.py))

**Purpose**: Split documents into semantic chunks for better retrieval

**Strategies**:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `SectionChunker` | Splits by markdown headers (`#`, `##`, etc.) | Structured docs with headers |
| `ParagraphChunker` | Splits by paragraphs | Prose-heavy content |
| `SWChunker` | Fixed-size sliding window | Uniform chunk sizes |
| `LLMChunker` | LLM-based semantic chunking | Maximum semantic coherence |

**Configuration**: Set via `CHUNK_STRATEGY` env var

### 3. ChromaDB Vector Store ([db/chroma/client.py](rag_app/src/db/chroma/client.py))

**Purpose**: Store and search document embeddings

**Key Features**:
- Persistent storage on disk
- Cosine similarity search
- Metadata filtering by repository
- Automatic embedding generation

**Embedding Model**: `multi-qa-distilbert-cos-v1` (sentence-transformers)

**Methods**:
- `add_documents()` - Index chunks
- `search()` - Semantic search
- `delete_by_repo()` - Remove repo data

### 4. RAG Service ([backend/services/rag.py](rag_app/src/backend/services/rag.py))

**Purpose**: Orchestrate retrieval and generation

**Process**:
1. Search ChromaDB for relevant chunks
2. Build context from top-k results
3. Retrieve conversation history
4. Construct augmented prompt
5. Generate response via OpenAI
6. Update conversation memory

**Context Window**: Top 5 chunks (configurable)

### 5. Conversation Memory ([core/memory.py](rag_app/src/core/memory.py))

**Purpose**: Maintain short-term conversation history

**Features**:
- In-memory storage (per session)
- Automatic pruning (keeps last 20 messages)
- Session-based isolation
- Metadata tracking (sources, tokens)

**Trade-off**: Memory is volatile (cleared on restart). Can be extended to use PostgreSQL for persistence.

### 6. Ingestion Service ([backend/services/ingestion.py](rag_app/src/backend/services/ingestion.py))

**Purpose**: End-to-end ingestion pipeline orchestration

**Pipeline**:
1. Validate & check for duplicates
2. Load repository from GitHub
3. Apply chunking strategy
4. Generate embeddings & index
5. Update PostgreSQL metadata
6. Handle errors gracefully

### 7. Streamlit Frontend ([frontend/app.py](rag_app/src/frontend/app.py))

**Purpose**: User interface for repo management and chat

**Features**:
- Repository ingestion UI
- Multi-select repository filter
- Chat interface with message history
- Source citations with similarity scores
- Session management
- Conversation clearing

## 🗄️ Database Schema

### PostgreSQL

```sql
-- repositories table
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    url VARCHAR(500) UNIQUE NOT NULL,
    owner VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    file_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    ingestion_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Status Values**: `pending`, `processing`, `completed`, `failed`

### ChromaDB

```python
# Collection metadata
{
    "hnsw:space": "cosine"  # Cosine similarity for search
}

# Document metadata structure
{
    "repo_url": "https://github.com/...",
    "chunk_index": 0,
    "filename": "path/to/file.md"
}
```

## 🔌 API Endpoints

### Repository Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/repository/ingest` | POST | Ingest a new repository |
| `/repository/list` | GET | List all repositories |
| `/repository/delete` | DELETE | Delete a repository |

### Chat & Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/search` | POST | Semantic search |
| `/chat/chat` | POST | Chat with RAG |
| `/chat/conversation/{id}` | GET | Get conversation |
| `/chat/conversation/{id}` | DELETE | Clear conversation |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/` | GET | API info |

## 🎨 Chunking Strategy Comparison

### Example: README.md

**Original**:
```markdown
# Project Title

This is the introduction.

## Installation

Run `pip install package`

## Usage

Import the library:
\`\`\`python
import package
\`\`\`
```

**Section Chunker** (level=2):
```
Chunk 1: "## Installation\n\nRun `pip install package`"
Chunk 2: "## Usage\n\nImport the library:..."
```

**Paragraph Chunker**:
```
Chunk 1: "# Project Title"
Chunk 2: "This is the introduction."
Chunk 3: "## Installation"
Chunk 4: "Run `pip install package`"
...
```

**Sliding Window** (size=100, overlap=20):
```
Chunk 1: chars 0-100
Chunk 2: chars 80-180
Chunk 3: chars 160-260
...
```

## 🚀 Deployment

### Docker Compose (Production)

```yaml
services:
  postgres:     # PostgreSQL database
  backend:      # FastAPI API server
  frontend:     # Streamlit UI
```

**Volumes**:
- `./data/postgres` → PostgreSQL data
- `./data/chroma` → ChromaDB embeddings

**Networks**: All services on `rag_network` bridge

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | Model name | `gpt-4o-mini` |
| `CHUNK_STRATEGY` | Chunking method | `section` |
| `SECTION_LEVEL` | Header level for sections | `2` |
| `DATABASE_URL` | PostgreSQL connection | Auto-configured |

## 🔍 Search & Retrieval

### Semantic Search Pipeline

1. **Query Embedding**: User query → sentence-transformers → 768-dim vector
2. **Vector Search**: ChromaDB cosine similarity search
3. **Filtering**: Optional repo_url filter
4. **Ranking**: Top-k by similarity (default k=5)
5. **Metadata**: Return content + filename + repo + distance

### RAG Prompt Template

```
Context from repositories:

[Source 1]
Repository: https://github.com/...
File: path/to/file.md
Content:
<chunk content>
---

[Source 2]
...

User Question: <query>

Please answer using the provided context...
```

## 📊 Performance Considerations

### Embedding Generation
- **Model**: multi-qa-distilbert-cos-v1 (~250MB)
- **Speed**: ~100 chunks/sec on CPU
- **Optimization**: Batch encoding, GPU support

### Vector Search
- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Complexity**: O(log n) query time
- **Scalability**: Efficient for 100k+ chunks

### Database Queries
- **PostgreSQL**: Indexed on `url`, `id`
- **ChromaDB**: Indexed by default

## 🛡️ Security Considerations

### Current Implementation
- No authentication (simplified)
- CORS enabled for all origins
- Direct OpenAI API access from backend

### Production Recommendations
1. Add JWT authentication
2. Restrict CORS to frontend domain
3. Rate limiting on API endpoints
4. API key rotation
5. Input validation & sanitization
6. PostgreSQL connection pooling

## 🧪 Testing Strategy

### Unit Tests
- Test chunking strategies with sample docs
- Validate database models
- Mock ChromaDB for service tests

### Integration Tests
- End-to-end ingestion pipeline
- RAG search and response generation
- API endpoint testing

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Ingest repo
curl -X POST http://localhost:8000/repository/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/username/repo"}'

# Search
curl -X POST http://localhost:8000/chat/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How to install?", "n_results": 3}'
```

## 📈 Future Enhancements

### Phase 1: Core Improvements
- [ ] Hybrid search (BM25 + semantic)
- [ ] Persistent conversation storage
- [ ] Support for code files (.py, .js, etc.)
- [ ] PDF document support

### Phase 2: Advanced Features
- [ ] User authentication & multi-tenancy
- [ ] Query rewriting & clarification
- [ ] Answer evaluation metrics
- [ ] Repository update detection

### Phase 3: Scale & Optimization
- [ ] Distributed ChromaDB
- [ ] Caching layer (Redis)
- [ ] Async ingestion with Celery
- [ ] Monitoring & observability

## 🔧 Troubleshooting

### Common Issues

**Issue**: Backend can't connect to PostgreSQL
**Solution**: Ensure PostgreSQL health check passes, check DATABASE_URL

**Issue**: ChromaDB permission errors
**Solution**: Ensure `data/chroma` directory exists with proper permissions

**Issue**: Slow ingestion
**Solution**: Use section/paragraph chunking instead of LLM chunking

**Issue**: Out of memory
**Solution**: Reduce chunk overlap, limit n_results, use smaller embedding model

## 📚 References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/docs/)
