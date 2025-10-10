from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Repository(Base):
    """Repository model to track ingested repositories."""

    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(500), unique=True, index=True, nullable=False)
    owner = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    file_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    ingestion_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)