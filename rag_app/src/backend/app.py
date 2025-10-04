from fastapi import FastAPI

from rag_app.backend.routers import health, search, agent


app = FastAPI(title="RAG App", version="0.1.0")

# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])
