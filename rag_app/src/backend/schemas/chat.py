"""Pydantic schemas for chat operations."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Schema for search request."""

    query: str = Field(..., description="Search query")
    repo_urls: list[str] | None = Field(None, description="Optional list of repository URLs to filter")
    n_results: int = Field(5, ge=1, le=20, description="Number of results to return")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I use React hooks?",
                "repo_urls": ["https://github.com/facebook/react"],
                "n_results": 5
            }
        }


class SearchResult(BaseModel):
    """Schema for individual search result."""

    content: str
    metadata: dict
    distance: float | None = None
    id: str | None = None


class ChatRequest(BaseModel):
    """Schema for chat request."""

    query: str = Field(..., description="User question")
    session_id: str = Field(..., description="Session ID for conversation tracking")
    repo_urls: list[str] | None = Field(None, description="Optional list of repository URLs to search")
    n_results: int = Field(5, ge=1, le=20, description="Number of context chunks to use")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I implement authentication?",
                "session_id": "user-123-session-456",
                "repo_urls": ["https://github.com/username/repo"],
                "n_results": 5
            }
        }


class Source(BaseModel):
    """Schema for source information."""

    index: int
    repo_url: str
    filename: str
    distance: float | None = None


class ChatResponse(BaseModel):
    """Schema for chat response."""

    answer: str
    sources: list[Source]
    context_used: int
    session_id: str
    error: str | None = None


class ConversationMessage(BaseModel):
    """Schema for conversation message."""

    role: str
    content: str
    metadata: dict = {}
