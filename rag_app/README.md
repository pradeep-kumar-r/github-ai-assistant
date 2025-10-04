# RAG App (Full-Stack)

A minimal full-stack scaffold for an agentic RAG application with a Streamlit frontend and a FastAPI backend.

This README documents how the app is laid out, what it depends on, and how to set it up locally based on the current repository structure.

## Overview

- **Frontend**: `streamlit`
- **Backend**: `fastapi`
- **Core libs**: `minsearch` (search), `openai` (LLM), `pydantic-ai` (agent), `sentence-transformers` (embeddings optional), `pandas`
- **Python**: 3.10+

See `rag_app/pyproject.toml` for the authoritative dependency list.

## Directory structure

```
rag_app/
├─ .python-version
├─ README.md
├─ main.py
├─ pyproject.toml
├─ config/
│  ├─ __init__.py
│  └─ settings.py
├─ backend/
│  ├─ __init__.py
│  ├─ app.py                 # FastAPI app entry (app = FastAPI())
│  ├─ routers/
│  │  ├─ __init__.py
│  │  ├─ health.py
│  │  ├─ search.py
│  │  └─ agent.py
│  ├─ schemas/
│  │  ├─ __init__.py
│  │  ├─ search.py
│  │  └─ agent.py
│  └─ services/
│     ├─ __init__.py
│     └─ retrieval.py
├─ frontend/
│  ├─ __init__.py
│  └─ app.py                 # Streamlit UI (def main())
├─ core/
│  ├─ __init__.py
│  ├─ ingest.py
│  ├─ chunk.py
│  ├─ index.py               # (placeholder)
│  ├─ search_lexical.py      # (placeholder)
│  ├─ search_vector.py       # (optional placeholder)
│  └─ agent.py               # (placeholder)
└─ db/
   ├─ __init__.py
   ├─ session.py             # (placeholder SQLite wiring)
   ├─ chroma/                # (optional vector backend)
   └─ sqllite/               # existing folder
```

## Prerequisites

- Python 3.10+
- Package manager: `uv` (recommended)

## Setup

From the repo root:

```bash
pip install uv
cd rag_app
uv sync
```

This installs dependencies declared in `rag_app/pyproject.toml`:

- `minsearch`, `openai`, `pydantic-ai`, `python-frontmatter`, `streamlit`, `sentence-transformers`, `fastapi`, `pydantic`, `pandas`

## Environment configuration

Create a `.env` file at the repo root or export variables in your shell:

```bash
# required for OpenAI client
export OPENAI_API_KEY="your-openai-key"
```

If you prefer a file, add `.env` (and do not commit secrets).

## Running (after implementing entry points)

- Backend (FastAPI): `rag_app/backend/app.py` exposes a FastAPI app called `app` and includes `/health`, `/search`, `/agent` routers (placeholders).

```bash
uv run uvicorn rag_app.backend.app:app --reload --port 8000
```

- Frontend (Streamlit): `rag_app/frontend/app.py` contains a minimal Streamlit app.

```bash
uv run streamlit run rag_app/frontend/app.py
```

- CLI (optional): `rag_app/main.py` currently prints a placeholder. Extend `main()` for CLI usage.

```bash
uv run python rag_app/main.py
```

## Development notes

- **Backend contracts**: HTTP routes live under `rag_app/backend/routers/` (e.g., `/search`, `/agent/ask`).
- **Core logic**: Put ingestion, indexing, search tools, and agent initialization under `rag_app/core/` and import them from backend/frontend.
- **State**: Start with `minsearch` in-memory index; persist to disk or external store if needed later.
- **Models**: Default to text search first; add embeddings (`sentence-transformers`) when necessary.
- **Secrets**: Use environment variables; never commit keys.

## Vector backend

- **MinSearch only (default)**: simplest lexical search, zero extra setup. Good for small corpora or when queries match document wording.
- **Chroma (optional)**: add semantic vector search with on-disk persistence. If you enable it, add `chromadb` to dependencies and configure a storage path via `CHROMA_DIR`. Use a small adapter in `core/search_vector.py` and combine with lexical results in `backend/services/retrieval.py`.

## Suggested next steps

- Flesh out routes in `rag_app/backend/routers/` for search and agent answers.
- Build `Streamlit` UI in `rag_app/frontend/app.py` and wire it to backend endpoints.
- Add ingestion/indexing utilities in `rag_app/core/` (download, chunk, index).
- Add basic evaluations and logging before deploying.

## Troubleshooting

- If `uv` is not found, ensure it’s installed (`pip install uv`) and your PATH is configured.
- `ImportError`: run commands from repo root so module paths like `rag_app.backend.api` resolve.
- Missing `OPENAI_API_KEY`: set the environment variable before running backend/agent code.

