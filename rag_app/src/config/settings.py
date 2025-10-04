import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Typed application settings loaded from environment variables.
    Add new fields as the app grows. Keep secrets in env, not in VCS.
    """

    # LLM/API
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Databases (placeholders; set when wiring persistence)
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    vector_backend: str = Field(default_factory=lambda: os.getenv("VECTOR_BACKEND", ""))

    # App
    env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))


settings = Settings()
