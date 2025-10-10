"""FastAPI backend application for RAG system."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..db.postgres import init_db
from ..logger import logger
from .routers import chat, health, repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting RAG backend application...")
    init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down RAG backend application...")


app = FastAPI(
    title="RAG App API",
    description="API for GitHub repository indexing and RAG-based Q&A",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(repository.router, prefix="/repository", tags=["repository"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG App API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "repository": "/repository",
            "chat": "/chat"
        }
    }
