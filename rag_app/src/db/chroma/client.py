import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ...logger import logger


class ChromaVectorStore:
    """ChromaDB vector store for embeddings and semantic search."""

    def __init__(self, persist_directory: str = "/app/data/chroma", collection_name: str = "rag_documents"):
        """Initialize ChromaDB client.

        Args:
            persist_directory: Path to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
        logger.info(f"Initialized embedding model: multi-qa-distilbert-cos-v1")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized with collection: {collection_name}")

    def add_documents(self, chunks: list[dict], repo_url: str) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'content' and metadata
            repo_url: Source repository URL for metadata

        Returns:
            Number of documents added
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return 0

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            chunk_id = f"{repo_url}_{i}"

            documents.append(content)
            ids.append(chunk_id)
            metadatas.append({
                'repo_url': repo_url,
                'chunk_index': chunk.get('index', i),
                'filename': chunk.get('filename', 'unknown'),
                **{k: str(v) for k, v in chunk.items() if k not in ['content', 'index']}
            })

        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return len(documents)

    def search(self, query: str, n_results: int = 5, filter_metadata: dict | None = None) -> list[dict]:
        """Search for similar documents using semantic search.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with content and metadata
        """
        where = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i] if results['ids'] else None
                })

        logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
        return formatted_results

    def delete_by_repo(self, repo_url: str) -> int:
        """Delete all documents from a specific repository.

        Args:
            repo_url: Repository URL to filter by

        Returns:
            Number of documents deleted
        """
        # Get all documents for this repo
        results = self.collection.get(
            where={"repo_url": repo_url}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} documents from repo: {repo_url}")
            return len(results['ids'])

        logger.info(f"No documents found for repo: {repo_url}")
        return 0

    def count_documents(self) -> int:
        """Get total number of documents in the collection."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """Reset the entire collection (delete all data)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning(f"Reset collection: {self.collection_name}")
