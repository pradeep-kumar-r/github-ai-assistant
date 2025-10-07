# RAG Application - Setup Guide

A full-stack RAG (Retrieval-Augmented Generation) application for GitHub repository Q&A with Streamlit UI, FastAPI backend, ChromaDB vector store, and PostgreSQL.

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │ (Port 8501)
│  - Repo Manager │
│  - Chat Interface│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI API    │ (Port 8000)
│  - /repository  │
│  - /chat        │
└────┬───────┬────┘
     │       │
     ▼       ▼
┌─────────┐ ┌──────────┐
│PostgreSQL│ │ ChromaDB │
│  Repos   │ │ Vectors  │
└─────────┘ └──────────┘
```

## 📋 Prerequisites

- Docker & Docker Compose
- OpenAI API Key

## 🚀 Quick Start

### 1. Clone and Navigate

```bash
cd rag_app
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

### 3. Create Data Directories

```bash
mkdir -p data/postgres data/chroma
```

### 4. Start the Application

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 5. Access the Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000

## 📖 Usage

### Indexing Repositories

1. Open Streamlit UI at http://localhost:8501
2. Enter a GitHub repository URL in the sidebar
3. Click "Index Repository"
4. Wait for ingestion to complete

### Asking Questions

1. Select one or more indexed repositories from the sidebar
2. Type your question in the chat input
3. View the AI response with sources

### Clearing Conversation

Click "Clear Conversation" in the sidebar to start a new chat session.

## 🔧 Configuration

### Chunking Strategies

Edit `.env` to change chunking strategy:

- `CHUNK_STRATEGY=section` - Splits by markdown headers (default)
- `CHUNK_STRATEGY=paragraph` - Splits by paragraphs
- `CHUNK_STRATEGY=sliding_window` - Fixed-size chunks with overlap
- `CHUNK_STRATEGY=llm` - LLM-based semantic chunking (slower, more expensive)

### Section Level

For section chunking, set header level:

```env
SECTION_LEVEL=2  # Split on ## headers
```

### OpenAI Model

```env
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o, gpt-3.5-turbo
```

## 🛠️ Development

### Running Locally (Without Docker)

1. Install dependencies:

```bash
pip install -e .
```

2. Start PostgreSQL (via Docker or local):

```bash
docker run -d \
  --name postgres \
  -e POSTGRES_USER=rag_user \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  postgres:15-alpine
```

3. Set environment variables:

```bash
export OPENAI_API_KEY=your_key_here
export DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db
export CHROMA_PERSIST_DIR=./data/chroma
```

4. Start backend:

```bash
uvicorn src.backend.app:app --reload --host 0.0.0.0 --port 8000
```

5. Start frontend (in another terminal):

```bash
export BACKEND_URL=http://localhost:8000
streamlit run src/frontend/app.py
```

## 🗂️ Data Persistence

All data is persisted in the `data/` directory:

- `data/postgres/` - PostgreSQL database
- `data/chroma/` - ChromaDB vector embeddings

To reset all data:

```bash
docker-compose down
rm -rf data/postgres data/chroma
mkdir -p data/postgres data/chroma
docker-compose up -d
```

## 🐛 Troubleshooting

### Backend won't start

Check logs:

```bash
docker-compose logs backend
```

Ensure OpenAI API key is set in `.env`.

### Database connection errors

Wait for PostgreSQL to be ready:

```bash
docker-compose logs postgres
```

### Ingestion fails

- Verify GitHub repository URL is valid
- Check repository has `.md` or `.mdx` files
- Review backend logs for detailed errors

## 📚 API Endpoints

### Repository Management

- `POST /repository/ingest` - Index a new repository
- `GET /repository/list` - List all indexed repositories
- `DELETE /repository/delete` - Delete a repository

### Chat & Search

- `POST /chat/search` - Semantic search across repositories
- `POST /chat/chat` - Chat with RAG context
- `GET /chat/conversation/{session_id}` - Get conversation history
- `DELETE /chat/conversation/{session_id}` - Clear conversation

### Health

- `GET /health` - Health check

## 🔄 Stopping the Application

```bash
# Stop services
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```

## 📝 Notes

- First run will download models (~500MB for sentence-transformers)
- Indexing large repositories may take several minutes
- ChromaDB data is stored locally (no cloud sync)
- Conversation history is in-memory (cleared on restart)

## 🎯 Next Steps

- Add user authentication
- Implement persistent conversation storage
- Add support for more file types (code files, PDFs)
- Implement hybrid search (lexical + semantic)
- Add evaluation metrics
