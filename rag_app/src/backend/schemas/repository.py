"""Pydantic schemas for repository operations."""

from pydantic import BaseModel, Field


class RepositoryIngest(BaseModel):
    """Schema for repository ingestion request."""

    url: str = Field(..., description="GitHub repository URL")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://github.com/username/repository"
            }
        }


class RepositoryInfo(BaseModel):
    """Schema for repository information response."""

    id: int
    url: str
    owner: str
    name: str
    file_count: int
    chunk_count: int
    status: str
    error: str | None = None
    created_at: str | None = None


class RepositoryDelete(BaseModel):
    """Schema for repository deletion request."""

    url: str = Field(..., description="GitHub repository URL to delete")


class IngestionResult(BaseModel):
    """Schema for ingestion result."""

    status: str
    message: str
    repo_id: int | None = None
    file_count: int | None = None
    chunk_count: int | None = None
    indexed_count: int | None = None
