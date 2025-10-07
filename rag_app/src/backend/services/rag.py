"""RAG service for semantic search and chat with context."""

from ...config.settings import settings
from ...core.llm import OpenAILLM
from ...core.memory import conversation_memory
from ...db.chroma.client import ChromaVectorStore
from ...logger import logger


class RAGService:
    """Service for RAG-based question answering."""

    def __init__(self, vector_store: ChromaVectorStore):
        """Initialize RAG service.

        Args:
            vector_store: ChromaDB vector store instance
        """
        self.vector_store = vector_store
        self.llm = OpenAILLM(api_key=settings.openai_api_key)

        # Setup LLM with system prompt
        system_prompt = """You are a helpful AI assistant that answers questions about code repositories.
You have access to documentation and code from GitHub repositories.

When answering:
1. Use the provided context from the repositories to answer questions
2. Be specific and cite relevant information when possible
3. If you don't know something or the context doesn't contain the answer, say so
4. Provide code examples when relevant
5. Keep answers clear and concise"""

        self.llm.setup(
            model=settings.openai_model,
            system_prompt=system_prompt
        )

    def search(self, query: str, repo_urls: list[str] | None = None, n_results: int = 5) -> list[dict]:
        """Search for relevant chunks.

        Args:
            query: Search query
            repo_urls: Optional list of repo URLs to filter by
            n_results: Number of results to return

        Returns:
            List of search results
        """
        # Build metadata filter if repos specified
        filter_metadata = None
        if repo_urls and len(repo_urls) > 0:
            # ChromaDB filters: for multiple repos, we need to search each separately
            # or use $or operator if supported
            if len(repo_urls) == 1:
                filter_metadata = {"repo_url": repo_urls[0]}

        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # If multiple repos, filter results post-search
        if repo_urls and len(repo_urls) > 1:
            results = [r for r in results if r.get('metadata', {}).get('repo_url') in repo_urls]

        return results

    def chat(
        self,
        query: str,
        session_id: str,
        repo_urls: list[str] | None = None,
        n_results: int = 5
    ) -> dict:
        """Chat with RAG context.

        Args:
            query: User question
            session_id: Session ID for conversation memory
            repo_urls: Optional list of repo URLs to search
            n_results: Number of search results to use as context

        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Step 1: Search for relevant context
            search_results = self.search(query, repo_urls, n_results)

            # Step 2: Build context from search results
            context_parts = []
            sources = []

            for i, result in enumerate(search_results):
                content = result.get('content', '')
                metadata = result.get('metadata', {})

                context_parts.append(f"[Source {i+1}]")
                context_parts.append(f"Repository: {metadata.get('repo_url', 'unknown')}")
                context_parts.append(f"File: {metadata.get('filename', 'unknown')}")
                context_parts.append(f"Content:\n{content}")
                context_parts.append("---")

                sources.append({
                    "index": i + 1,
                    "repo_url": metadata.get('repo_url', 'unknown'),
                    "filename": metadata.get('filename', 'unknown'),
                    "distance": result.get('distance')
                })

            context = "\n".join(context_parts)

            # Step 3: Get conversation history
            history = conversation_memory.get_messages_for_llm(session_id)

            # Step 4: Build prompt with context
            prompt = f"""Context from repositories:

{context}

User Question: {query}

Please answer the question using the provided context. If the context doesn't contain enough information to answer the question, say so."""

            # Step 5: Generate response
            _, response_text = self.llm.generate(prompt)

            # Step 6: Update conversation memory
            conversation_memory.add_message(
                session_id=session_id,
                role="user",
                content=query,
                metadata={"sources_used": len(sources)}
            )

            conversation_memory.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
                metadata={"sources": sources}
            )

            logger.info(f"Generated answer for session {session_id}")

            return {
                "answer": response_text,
                "sources": sources,
                "context_used": len(search_results),
                "session_id": session_id
            }

        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "context_used": 0,
                "session_id": session_id,
                "error": str(e)
            }

    def get_conversation_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of messages
        """
        return conversation_memory.get_history(session_id)

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session.

        Args:
            session_id: Session ID
        """
        conversation_memory.clear_session(session_id)
        logger.info(f"Cleared conversation for session {session_id}")
