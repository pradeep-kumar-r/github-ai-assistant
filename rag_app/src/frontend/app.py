import streamlit as st


def main():
    st.set_page_config(page_title="RAG App", page_icon="🤖", layout="centered")
    st.title("🤖 RAG App UI")
    st.caption("Streamlit frontend for the agentic RAG application")

    st.markdown(
        """
        - Use this UI to interact with the backend once endpoints are implemented.
        - Start the backend (FastAPI) separately and wire the UI to its routes (e.g., /health, /search, /agent).
        """
    )


if __name__ == "__main__":
    main()

