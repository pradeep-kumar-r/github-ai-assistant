"""Repository management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...config.settings import settings
from ...db.chroma.client import ChromaVectorStore
from ...db.database import get_db
from ..schemas.repository import IngestionResult, RepositoryDelete, RepositoryIngest, RepositoryInfo
from ..services.ingestion import IngestionService

router = APIRouter()

# Initialize ChromaDB (singleton)
_vector_store = None


def get_vector_store() -> ChromaVectorStore:
    """Get or create ChromaDB vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name
        )
    return _vector_store


@router.post("/ingest", response_model=IngestionResult)
async def ingest_repository(
    request: RepositoryIngest,
    db: Session = Depends(get_db),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Ingest a GitHub repository.

    Downloads the repository, chunks the markdown files, and indexes them in ChromaDB.
    """
    ingestion_service = IngestionService(db, vector_store)
    result = await ingestion_service.ingest_repository(request.url)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


@router.get("/list", response_model=list[RepositoryInfo])
async def list_repositories(
    db: Session = Depends(get_db),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Get list of all indexed repositories."""
    ingestion_service = IngestionService(db, vector_store)
    repositories = ingestion_service.get_all_repositories()
    return repositories


@router.delete("/delete", response_model=dict)
async def delete_repository(
    request: RepositoryDelete,
    db: Session = Depends(get_db),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Delete a repository and its indexed data."""
    ingestion_service = IngestionService(db, vector_store)
    result = ingestion_service.delete_repository(request.url)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    elif result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])

    return result
