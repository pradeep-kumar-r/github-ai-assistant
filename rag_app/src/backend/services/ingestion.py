"""Ingestion service to orchestrate data loading, chunking, and indexing."""

from sqlalchemy.orm import Session

from ...config.settings import settings
from ...core.chunking import LLMChunker, ParagraphChunker, SectionChunker, SWChunker
from ...core.ingestion import DataLoader
from ...core.llm import OpenAILLM
from ...db.chroma.client import ChromaVectorStore
from ...db.models import Repository
from ...logger import logger


class IngestionService:
    """Service to handle repository ingestion pipeline."""

    def __init__(self, db: Session, vector_store: ChromaVectorStore):
        """Initialize ingestion service.

        Args:
            db: Database session
            vector_store: ChromaDB vector store instance
        """
        self.db = db
        self.vector_store = vector_store
        self.data_loader = DataLoader()

    def get_chunker(self):
        """Get chunker instance based on settings."""
        strategy = settings.chunk_strategy.lower()

        if strategy == "section":
            return SectionChunker(level=settings.section_level)
        elif strategy == "paragraph":
            return ParagraphChunker()
        elif strategy == "sliding_window":
            return SWChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
        elif strategy == "llm":
            llm = OpenAILLM(api_key=settings.openai_api_key)
            llm.setup(model=settings.openai_model)
            return LLMChunker(llm=llm)
        else:
            logger.warning(f"Unknown chunk strategy: {strategy}, using section chunker")
            return SectionChunker(level=2)

    async def ingest_repository(self, repo_url: str) -> dict:
        """Ingest a GitHub repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Dictionary with ingestion results
        """
        # Check if repo already exists
        existing_repo = self.db.query(Repository).filter(Repository.url == repo_url).first()

        if existing_repo:
            if existing_repo.ingestion_status == "completed":
                return {
                    "status": "already_indexed",
                    "message": f"Repository already indexed: {repo_url}",
                    "repo_id": existing_repo.id,
                    "file_count": existing_repo.file_count,
                    "chunk_count": existing_repo.chunk_count
                }
            else:
                # Update existing record
                repo = existing_repo
                repo.ingestion_status = "processing"
                self.db.commit()
        else:
            # Parse URL to get owner and name
            repo_owner, repo_name = DataLoader._parse_url(repo_url)

            # Create new repository record
            repo = Repository(
                url=repo_url,
                owner=repo_owner,
                name=repo_name,
                ingestion_status="processing"
            )
            self.db.add(repo)
            self.db.commit()
            self.db.refresh(repo)

        try:
            # Step 1: Load repository
            logger.info(f"Loading repository: {repo_url}")
            repo_data = self.data_loader.load_repo_from_url(repo_url)
            file_count = len(repo_data)

            if file_count == 0:
                raise ValueError("No markdown files found in repository")

            # Step 2: Chunk documents
            logger.info(f"Chunking {file_count} documents")
            chunker = self.get_chunker()
            all_chunks = []

            for doc in repo_data:
                content = doc.get('content', '')
                if content:
                    chunks = chunker.chunk(content)
                    # Add document metadata to each chunk
                    for chunk in chunks:
                        chunk['filename'] = doc.get('filename', 'unknown')
                        chunk['repo_url'] = repo_url
                    all_chunks.extend(chunks)

            chunk_count = len(all_chunks)
            logger.info(f"Created {chunk_count} chunks")

            # Step 3: Index in ChromaDB
            logger.info(f"Indexing chunks in ChromaDB")
            indexed_count = self.vector_store.add_documents(all_chunks, repo_url)

            # Step 4: Update repository status
            repo.file_count = file_count
            repo.chunk_count = chunk_count
            repo.ingestion_status = "completed"
            repo.error_message = None
            self.db.commit()

            logger.info(f"Successfully ingested repository: {repo_url}")

            return {
                "status": "success",
                "message": f"Successfully indexed repository: {repo_url}",
                "repo_id": repo.id,
                "file_count": file_count,
                "chunk_count": chunk_count,
                "indexed_count": indexed_count
            }

        except Exception as e:
            logger.error(f"Failed to ingest repository {repo_url}: {str(e)}")

            # Update repository with error
            repo.ingestion_status = "failed"
            repo.error_message = str(e)
            self.db.commit()

            return {
                "status": "error",
                "message": f"Failed to index repository: {str(e)}",
                "repo_id": repo.id
            }

    def get_all_repositories(self) -> list[dict]:
        """Get all indexed repositories.

        Returns:
            List of repository dictionaries
        """
        repos = self.db.query(Repository).all()
        return [
            {
                "id": repo.id,
                "url": repo.url,
                "owner": repo.owner,
                "name": repo.name,
                "file_count": repo.file_count,
                "chunk_count": repo.chunk_count,
                "status": repo.ingestion_status,
                "error": repo.error_message,
                "created_at": repo.created_at.isoformat() if repo.created_at else None
            }
            for repo in repos
        ]

    def delete_repository(self, repo_url: str) -> dict:
        """Delete a repository and its indexed data.

        Args:
            repo_url: Repository URL to delete

        Returns:
            Dictionary with deletion results
        """
        repo = self.db.query(Repository).filter(Repository.url == repo_url).first()

        if not repo:
            return {
                "status": "not_found",
                "message": f"Repository not found: {repo_url}"
            }

        try:
            # Delete from vector store
            deleted_count = self.vector_store.delete_by_repo(repo_url)

            # Delete from database
            self.db.delete(repo)
            self.db.commit()

            return {
                "status": "success",
                "message": f"Deleted repository: {repo_url}",
                "chunks_deleted": deleted_count
            }

        except Exception as e:
            logger.error(f"Failed to delete repository {repo_url}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to delete repository: {str(e)}"
            }
