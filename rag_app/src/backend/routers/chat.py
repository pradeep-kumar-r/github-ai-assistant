"""Chat and search endpoints."""

from fastapi import APIRouter, Depends

from ...config.settings import settings
from ...db.chroma.client import ChromaVectorStore
from ..schemas.chat import ChatRequest, ChatResponse, ConversationMessage, SearchRequest, SearchResult
from ..services.rag import RAGService

router = APIRouter()

# Initialize ChromaDB (singleton)
_vector_store = None
_rag_service = None


def get_vector_store() -> ChromaVectorStore:
    """Get or create ChromaDB vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name
        )
    return _vector_store


def get_rag_service(vector_store: ChromaVectorStore = Depends(get_vector_store)) -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(vector_store)
    return _rag_service


@router.post("/search", response_model=list[SearchResult])
async def search(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Search for relevant chunks across indexed repositories."""
    results = rag_service.search(
        query=request.query,
        repo_urls=request.repo_urls,
        n_results=request.n_results
    )
    return results


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Chat with RAG context from indexed repositories."""
    response = rag_service.chat(
        query=request.query,
        session_id=request.session_id,
        repo_urls=request.repo_urls,
        n_results=request.n_results
    )
    return response


@router.get("/conversation/{session_id}", response_model=list[ConversationMessage])
async def get_conversation(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get conversation history for a session."""
    history = rag_service.get_conversation_history(session_id)
    return history


@router.delete("/conversation/{session_id}")
async def clear_conversation(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Clear conversation history for a session."""
    rag_service.clear_conversation(session_id)
    return {"status": "success", "message": f"Cleared conversation for session {session_id}"}
