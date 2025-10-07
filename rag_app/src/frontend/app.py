"""Streamlit frontend for RAG application."""

import os
import uuid

import httpx
import streamlit as st

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_repos" not in st.session_state:
        st.session_state.selected_repos = []


def get_repositories():
    """Fetch list of indexed repositories."""
    try:
        response = httpx.get(f"{BACKEND_URL}/repository/list", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch repositories: {str(e)}")
        return []


def ingest_repository(repo_url: str):
    """Ingest a new repository."""
    try:
        with st.spinner(f"Ingesting repository: {repo_url}"):
            response = httpx.post(
                f"{BACKEND_URL}/repository/ingest",
                json={"url": repo_url},
                timeout=300  # 5 minutes timeout for ingestion
            )
            response.raise_for_status()
            result = response.json()
            return result
    except Exception as e:
        st.error(f"Failed to ingest repository: {str(e)}")
        return None


def send_chat_message(query: str, repo_urls: list[str]):
    """Send chat message to backend."""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/chat/chat",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "repo_urls": repo_urls,
                "n_results": 5
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to send message: {str(e)}")
        return None


def clear_conversation():
    """Clear conversation history."""
    try:
        response = httpx.delete(
            f"{BACKEND_URL}/chat/conversation/{st.session_state.session_id}",
            timeout=10
        )
        response.raise_for_status()
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
    except Exception as e:
        st.error(f"Failed to clear conversation: {str(e)}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    st.title("🤖 GitHub Repository Q&A Assistant")
    st.caption("Ask questions about your indexed GitHub repositories")

    # Sidebar for repository management
    with st.sidebar:
        st.header("📚 Repository Management")

        # Ingest new repository
        st.subheader("Add New Repository")
        new_repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repo",
            key="new_repo_input"
        )

        if st.button("Index Repository", type="primary", use_container_width=True):
            if new_repo_url:
                result = ingest_repository(new_repo_url)
                if result:
                    if result["status"] == "success":
                        st.success(f"✅ {result['message']}")
                        st.info(f"Files: {result['file_count']}, Chunks: {result['chunk_count']}")
                    elif result["status"] == "already_indexed":
                        st.warning(f"ℹ️ {result['message']}")
                    else:
                        st.error(f"❌ {result['message']}")
                    st.rerun()
            else:
                st.warning("Please enter a repository URL")

        st.divider()

        # List indexed repositories
        st.subheader("Indexed Repositories")
        repositories = get_repositories()

        if repositories:
            completed_repos = [r for r in repositories if r["status"] == "completed"]

            if completed_repos:
                st.session_state.selected_repos = st.multiselect(
                    "Select repositories to query",
                    options=[r["url"] for r in completed_repos],
                    default=st.session_state.selected_repos,
                    format_func=lambda x: f"{x.split('/')[-2]}/{x.split('/')[-1]}"
                )

                st.caption(f"📊 Total: {len(completed_repos)} repositories indexed")

                # Show repository details in expander
                with st.expander("View Details"):
                    for repo in completed_repos:
                        st.write(f"**{repo['owner']}/{repo['name']}**")
                        st.write(f"- Files: {repo['file_count']}")
                        st.write(f"- Chunks: {repo['chunk_count']}")
                        st.write(f"- Indexed: {repo['created_at'][:10]}")
                        st.divider()
            else:
                st.info("No successfully indexed repositories yet")

            # Show failed ingestions
            failed_repos = [r for r in repositories if r["status"] == "failed"]
            if failed_repos:
                with st.expander("⚠️ Failed Ingestions", expanded=False):
                    for repo in failed_repos:
                        st.error(f"**{repo['url']}**")
                        st.write(f"Error: {repo['error']}")
        else:
            st.info("No repositories indexed yet. Add one above!")

        st.divider()

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_conversation()
            st.rerun()

    # Main chat interface
    if not st.session_state.selected_repos:
        st.warning("👈 Please select at least one repository from the sidebar to start chatting")
    else:
        st.success(f"✅ Chatting with {len(st.session_state.selected_repos)} repository/repositories")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("📎 Sources", expanded=False):
                            for source in message["sources"]:
                                st.write(f"**Source {source['index']}**")
                                st.write(f"- Repository: {source['repo_url']}")
                                st.write(f"- File: {source['filename']}")
                                if source.get('distance') is not None:
                                    st.write(f"- Similarity: {1 - source['distance']:.2%}")
                                st.divider()

        # Chat input
        if prompt := st.chat_input("Ask a question about your repositories..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = send_chat_message(prompt, st.session_state.selected_repos)

                    if response:
                        answer = response.get("answer", "No response")
                        sources = response.get("sources", [])

                        st.markdown(answer)

                        # Show sources
                        if sources:
                            with st.expander("📎 Sources", expanded=False):
                                for source in sources:
                                    st.write(f"**Source {source['index']}**")
                                    st.write(f"- Repository: {source['repo_url']}")
                                    st.write(f"- File: {source['filename']}")
                                    if source.get('distance') is not None:
                                        st.write(f"- Similarity: {1 - source['distance']:.2%}")
                                    st.divider()

                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })


if __name__ == "__main__":
    main()

