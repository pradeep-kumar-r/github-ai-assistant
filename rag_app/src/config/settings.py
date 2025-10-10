import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Typed application settings loaded from environment variables."""

    # LLM/API
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # PostgreSQL Database
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")
    )

    # ChromaDB
    chroma_persist_dir: str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma"))
    chroma_collection_name: str = Field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"))

    # Chunking
    chunk_strategy: str = Field(
        default_factory=lambda: os.getenv("CHUNK_STRATEGY", "section")
    )  # section, paragraph, sliding_window
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "2000")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    section_level: int = Field(default_factory=lambda: int(os.getenv("SECTION_LEVEL", "2")))

    # App
    env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    backend_host: str = Field(default_factory=lambda: os.getenv("BACKEND_HOST", "0.0.0.0"))
    backend_port: int = Field(default_factory=lambda: int(os.getenv("BACKEND_PORT", "8000")))


settings = Settings()
