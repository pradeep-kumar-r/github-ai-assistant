# RAG App (Full-Stack)

An agentic RAG application that acts as a technical assistant for coding, software development, and programming questions. Built with a Streamlit frontend and FastAPI backend, it combines lexical and semantic search with LLM-powered agents.

## Overview

### Tech Stack

- **Frontend**: Streamlit 1.50+ (interactive UI)
- **Backend**: FastAPI 0.104+ (REST API)
- **AI Agent**: Pydantic AI 1.0.9 + LangGraph 0.1+ (agentic workflows)
- **LLM**: OpenAI GPT-4o-mini (via `openai>=1.108.2`)
- **Vector Search**: MinSearch 0.0.5+ (in-memory lexical search)
- **Embeddings**: Sentence-Transformers 2.2+ (semantic embeddings, optional)
- **Data Processing**: Pandas 2.2+, Python-Frontmatter 1.1+ (markdown parsing)
- **Logging**: Loguru 0.7.3+ (structured logging to console + file)
- **Python**: 3.10+

See `pyproject.toml` for the complete dependency list.

## Directory Structure

```
rag_app/
├─ .python-version           # Python version (3.10+)
├─ README.md
├─ main.py                   # CLI entry point
├─ pyproject.toml            # Dependencies & tool config (Ruff, etc.)
├─ uv.lock                   # Lockfile (uv package manager)
├─ data/                     # Ingested data & pickled repos
├─ logs/                     # Application logs (app.log, app_debug.log)
├─ tests/                    # Unit & integration tests
└─ src/
   ├─ __init__.py
   ├─ constants.py           # Global constants (ROOT_DIR, etc.)
   ├─ logger.py              # Loguru config (stdout + file)
   ├─ config/
   │  ├─ __init__.py
   │  └─ settings.py         # Pydantic settings (env vars)
   ├─ backend/
   │  ├─ __init__.py
   │  ├─ app.py              # FastAPI app entry
   │  ├─ routers/
   │  │  ├─ __init__.py
   │  │  ├─ health.py        # Health check endpoint
   │  │  ├─ search.py        # Search endpoint
   │  │  └─ agent.py         # Agent /ask endpoint
   │  ├─ schemas/
   │  │  ├─ __init__.py
   │  │  ├─ search.py        # Pydantic models for search
   │  │  └─ agent.py         # Pydantic models for agent
   │  └─ services/
   │     ├─ __init__.py
   │     └─ retrieval.py     # Hybrid search orchestration
   ├─ frontend/
   │  ├─ __init__.py
   │  └─ app.py              # Streamlit UI
   ├─ core/
   │  ├─ __init__.py
   │  ├─ ingestion.py        # GitHub repo loader (DataLoader)
   │  └─ chunk.py            # Sliding window chunking
   └─ db/
      ├─ __init__.py
      ├─ session.py          # DB session management
      ├─ chroma/             # ChromaDB vector store (optional)
      └─ sqllite/            # SQLite backend (optional)
```

## Architecture & Components

### 1. Data Ingestion (`src/core/ingestion.py`)

**Framework**: `requests`, `zipfile`, `python-frontmatter`

The `DataLoader` class handles:
- **GitHub Repository Loading**: Downloads repos as ZIP archives from `github.com` via `requests`
- **Markdown Parsing**: Extracts `.md` and `.mdx` files using `python-frontmatter` to parse YAML frontmatter
- **Validation**: URL validation and error handling with structured logging
- **Persistence**: Saves/loads repo data as pickled objects for reuse

**Key Methods**:
- `load_repo_from_url(url)`: Downloads and parses a GitHub repo
- `save_repo_data(path)`: Persists data to disk
- `load_existing_repo_data(path)`: Loads cached data

### 2. Chunking Strategy (`src/core/chunk.py`)

**Strategy**: Sliding window with overlap

- **Default**: 2000 characters per chunk, 1000 character step (50% overlap)
- **Purpose**: Maintains context across chunk boundaries for better retrieval
- **Output**: List of dicts with `start` position and `content`

**Function**: `sliding_window(text, size=2000, step=1000)`

### 3. Vector Search (`src/backend/services/retrieval.py`)

**Framework**: MinSearch (in-memory lexical search)

- **Current**: Lexical BM25-style search via MinSearch
- **Future**: Hybrid search combining lexical + semantic (Sentence-Transformers embeddings)
- **Storage**: In-memory index, optionally persisted to disk
- **Optional**: ChromaDB for persistent vector storage (`src/db/chroma/`)

**Function**: `hybrid_search(query, top_k=5)` (placeholder for hybrid retrieval)

### 4. AI Agent (`src/backend/routers/agent.py`)

**Framework**: Pydantic AI + LangGraph + OpenAI

- **LLM**: OpenAI GPT-4o-mini (configurable via `OPENAI_API_KEY`)
- **Agent Framework**: Pydantic AI for structured outputs, LangGraph for multi-step workflows
- **Tools**: Retrieval tool (search knowledge base), future: code execution, web search
- **Endpoint**: `POST /agent/ask` with conversation tracking

**Schemas**: `AskRequest`, `AskResponse` in `src/backend/schemas/agent.py`

### 5. Logging (`src/logger.py`)

**Framework**: Loguru

- **Console**: INFO level, colorized output to `stdout`
- **File (INFO)**: `logs/app.log` (rotation: 10 MB, retention: 14 days)
- **File (DEBUG)**: `logs/app_debug.log` (all debug messages)
- **Format**: Timestamp, level, module:function:line, message

### 6. Configuration (`src/config/settings.py`)

**Framework**: Pydantic BaseModel

Environment variables loaded via Pydantic:
- `OPENAI_API_KEY`: Required for LLM calls
- `DATABASE_URL`: Optional DB connection string
- `VECTOR_BACKEND`: Optional vector store selection
- `APP_ENV`: Environment (development/production)

## Prerequisites

- Python 3.10+
- Package manager: `uv` (recommended) or `pip`
- OpenAI API key

## Setup

### 1. Install Dependencies

From the repo root:

```bash
pip install uv
cd rag_app
uv sync
```

For development tools (Ruff, Black, isort, Loguru):
```bash
uv sync --extra dev
```

This installs all dependencies from `pyproject.toml`.

### 2. Environment Configuration

Create a `.env` file at the repo root or export variables:

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export DATABASE_URL="sqlite:///./rag_app.db"
export VECTOR_BACKEND="minsearch"  # or "chroma"
export APP_ENV="development"
```

**Security**: Never commit `.env` files or API keys to version control.

## Running the Application

### Backend (FastAPI)

Start the FastAPI server:

```bash
cd rag_app
uv run uvicorn src.backend.app:app --reload --port 8000
```

**Endpoints**:
- `GET /health` - Health check
- `POST /search` - Search knowledge base
- `POST /agent/ask` - Ask the AI agent

API docs available at: `http://localhost:8000/docs`

### Frontend (Streamlit)

Start the Streamlit UI:

```bash
cd rag_app
uv run streamlit run src/frontend/app.py
```

Access at: `http://localhost:8501`

### CLI (Optional)

Run the CLI entry point:

```bash
cd rag_app
uv run python main.py
```

Currently prints a placeholder. Extend `main()` for CLI-based ingestion or queries.

## Development Workflow

### Code Formatting & Linting

The project uses Ruff for formatting and linting:

```bash
# Format all Python files
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check . --fix

# Alternative: Black + isort (also available in dev dependencies)
uv run black .
uv run isort .
```

Configuration in `pyproject.toml`:
- Line length: 120
- Target: Python 3.10+
- Style: double quotes, space indentation

### Project Structure Guidelines

- **Backend routes**: Add new endpoints in `src/backend/routers/`
- **Schemas**: Define Pydantic models in `src/backend/schemas/`
- **Core logic**: Implement ingestion, chunking, search in `src/core/`
- **Services**: Orchestration layer in `src/backend/services/`
- **Configuration**: Add env vars to `src/config/settings.py`
- **Logging**: Import logger from `src.logger` (not relative imports)

### Data Storage Strategy

**Current**:
- **MinSearch**: In-memory lexical search (BM25-style)
- **Pickle**: Serialized repo data in `data/` directory
- **Logs**: Structured logs in `logs/` (INFO + DEBUG levels)

**Future**:
- **ChromaDB**: Persistent vector store for semantic search
- **SQLite**: Conversation history and user sessions
- **Hybrid Search**: Combine lexical (MinSearch) + semantic (embeddings)

### Vector Backend Options

#### MinSearch (Default)
- **Pros**: Zero setup, fast, good for exact keyword matches
- **Cons**: No semantic understanding, requires query-document term overlap
- **Use case**: Small corpora, technical documentation with consistent terminology

#### ChromaDB (Optional)
- **Pros**: Semantic search, persistent storage, handles synonyms/paraphrasing
- **Cons**: Requires embeddings generation, larger memory footprint
- **Setup**: Add `chromadb` to dependencies, configure `CHROMA_DIR` env var
- **Implementation**: Create adapter in `src/core/search_vector.py`

## Ingestion Pipeline

### Loading GitHub Repositories

```python
from src.core.ingestion import DataLoader

loader = DataLoader()
repo_data = loader.load_repo_from_url("https://github.com/owner/repo")
loader.save_repo_data("data/repo.pkl")
```

**Process**:
1. Validates GitHub URL
2. Downloads repo as ZIP from `codeload.github.com`
3. Extracts `.md` and `.mdx` files
4. Parses YAML frontmatter with `python-frontmatter`
5. Logs progress (files processed, errors)
6. Returns list of dicts with `filename`, `url`, `content`, metadata

### Chunking Documents

```python
from src.core.chunk import sliding_window

chunks = sliding_window(document_text, size=2000, step=1000)
# Returns: [{"start": 0, "content": "..."}, {"start": 1000, "content": "..."}, ...]
```

**Strategy**: 50% overlap ensures context continuity across chunks.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest --cov=src tests/
```

## Troubleshooting

### Common Issues

**ImportError: attempted relative import with no known parent package**
- **Cause**: Running a module with relative imports directly as a script
- **Fix**: Use absolute imports (`from src.module import X`) or run as module (`python -m src.module`)

**ModuleNotFoundError: No module named 'src'**
- **Cause**: Running commands from wrong directory
- **Fix**: Always run from `rag_app/` root directory

**Missing OPENAI_API_KEY**
- **Cause**: Environment variable not set
- **Fix**: `export OPENAI_API_KEY="sk-..."` or add to `.env` file

**uv command not found**
- **Cause**: `uv` not installed or not in PATH
- **Fix**: `pip install uv` and restart terminal

### Logs Location

- **Console**: INFO level, colorized
- **File (INFO)**: `logs/app.log`
- **File (DEBUG)**: `logs/app_debug.log`

Check logs for detailed error traces and debugging information.

## Next Steps

1. **Implement Agent**: Wire Pydantic AI + LangGraph in `src/backend/routers/agent.py`
2. **Add Search**: Implement MinSearch indexing in `src/core/`
3. **Build UI**: Create Streamlit interface in `src/frontend/app.py`
4. **Add Tests**: Unit tests for ingestion, chunking, retrieval
5. **Hybrid Search**: Combine lexical + semantic search with embeddings
6. **Conversation Memory**: Store chat history in SQLite
7. **Evaluation**: Add metrics for retrieval quality and agent accuracy

