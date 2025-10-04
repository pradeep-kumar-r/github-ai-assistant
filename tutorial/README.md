# Tutorials README

This folder contains hands-on notebooks that build a GitHub-powered AI assistant step by step. It mirrors Days 1–5 of the curriculum in `course_material/` and focuses on a minimal, clean implementation you can run locally.

## Overview

- **Goal**: Build an agent that answers questions about a GitHub repo using search over ingested docs.
- **Flow**:
  - Ingest repo docs (Day 1)
  - Chunk large docs (Day 2)
  - Add search (text/vector/hybrid) (Day 3)
  - Turn search into a tool and build an agent (Day 4)
  - Evaluate (logs, LLM-as-judge, simple IR metrics) (Day 5)

## Structure

- `tutorial/notebooks/01-download-repo.ipynb` — Download and parse GitHub repo markdown/MDX with frontmatter.
- `tutorial/notebooks/02-chunk-documents.ipynb` — Chunking: sliding window, sections, and notes on intelligent chunking.
- `tutorial/notebooks/03-search-chunked-document.ipynb` — Text, vector, hybrid search with `minsearch` and `sentence-transformers`.
- `tutorial/notebooks/04-build-agent-tools.ipynb` — Tool-calling with OpenAI; Pydantic AI agent using a `search(query)` tool.
- `tutorial/notebooks/05-evaluation.ipynb` — Logging, LLM-as-judge, structured outputs, metrics (hit-rate/MRR) basics.
- `tutorial/notebooks/05b-qgen-qna-evals.ipynb` — Optional: question generation and automated eval set creation. To illustrate an end to end pipelines.

See detailed background in `course_material/`:

- `Day 1_ Ingest and Index Your Data.md`
- `Day 2_ Chunking and Intelligent Processing for Data.md`
- `Day 3_ Add Search.md`
- `Day 4_ Agents and Tools.md`
- `Day 5_ Evaluation.md`

## Prerequisites

- Python 3.10+
- Package manager: `uv` (recommended). Alternative managers work, but examples reference `uv`.
- An LLM provider key (e.g., `OPENAI_API_KEY`) for agent and intelligent chunking. Store as an environment variable; never commit keys.

## Setup

From the project root (or a dedicated env):

```bash
pip install uv
uv init  # if starting fresh in a new folder
uv add requests python-frontmatter
uv add --dev jupyter
# Add as you progress:
uv add minsearch sentence-transformers pydantic-ai openai pandas tqdm streamlit
```

Start Jupyter:

```bash
uv run jupyter notebook
```

Set your API key (bash):

```bash
export OPENAI_API_KEY="your-key"
```

Optional: use `direnv` to auto-load environment variables. Ensure secrets are not committed.

## How to run the tutorials

1. Open each notebook in order under `tutorial/notebooks/`.
2. Execute cells top-to-bottom. Parameters like repo owner/name can be edited inline.
3. For vector search or intelligent chunking, run cells that require external models/keys only if needed (cost/latency trade-offs).

## Notebook guide

- **01-download-repo.ipynb**
  - Download a GitHub repository as a ZIP.
  - Parse `.md`/`.mdx` with `python-frontmatter` into records: `{metadata..., content, filename}`.
  - Handles path normalization to make later citation links easy.

- **02-chunk-documents.ipynb**
  - Sliding-window chunking (size/overlap) for long docs.
  - Alternatives: paragraph/section-based splitting; notes on LLM-assisted chunking.
  - Produces a list of smaller, search-friendly records.

- **03-search-chunked-document.ipynb**
  - Text search with `minsearch.Index` on fields like `content/title/filename`.
  - Vector search with `sentence-transformers` and `minsearch.VectorSearch`.
  - Hybrid approach: combine lexical and vector results; simple de-duplication.

- **04-build-agent-tools.ipynb**
  - OpenAI function-calling vs. library-managed tools.
  - Pydantic AI `Agent` with a `search(query)` tool wired to your index.
  - System prompt patterns: enforce search-first, retries, and citations.

- **05-evaluation.ipynb**
  - Logging full interactions to JSON for later analysis.
  - LLM-as-judge with structured outputs (Pydantic models) to evaluate answers.
  - Basic IR metrics sketch (hit rate, MRR) for search components.

- **05b-qgen-qna-evals.ipynb** (optional)
  - Generate realistic questions from your corpus with an LLM.
  - Batch-run your agent, log results, and evaluate at scale.

## Mapping to course days

- Day 1 → `01-download-repo.ipynb`
- Day 2 → `02-chunk-documents.ipynb`
- Day 3 → `03-search-chunked-document.ipynb`
- Day 4 → `04-build-agent-tools.ipynb`
- Day 5 → `05-evaluation.ipynb`, `05b-qgen-qna-evals.ipynb`

## Tips and cautions

- **Security/keys**: Keep API keys in env vars; never commit secrets.
- **Costs**: Intelligent chunking and evals incur LLM calls; start simple, then scale.
- **Simplicity**: Favor text search first; add vector/hybrid only when needed (measure!).
- **Citations**: Normalize filenames early so agents can link to exact GitHub paths.
- **Reproducibility**: Pin dependencies for long-lived demos; export `requirements.txt` if deploying Streamlit Cloud.

## Next steps

- Convert the notebooks into modules/scripts for productionization.
- Add a Streamlit UI and deploy; surface references and streamed responses.
- Track interaction logs and iterate on prompts/tools based on evaluation metrics.
- Build a full-stack app with a UI & a backend and deploy it to a server.

## References

- `course_material/Day 1_ Ingest and Index Your Data.md`
- `course_material/Day 2_ Chunking and Intelligent Processing for Data.md`
- `course_material/Day 3_ Add Search.md`
- `course_material/Day 4_ Agents and Tools.md`
- `course_material/Day 5_ Evaluation.md`

