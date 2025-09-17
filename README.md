# Customizable Retrieval-Augmented Generation workspace

A customisable Retrieval-Augmented Generation (RAG) workspace built with Streamlit. You can point it at *any* public documentation site, crawl and chunk the text, embed it locally or with OpenAI, and query through OpenAI or your local Ollama model. All functionalities from a single UI that now ships with a night-friendly theme.

---

## Customise It For Your Docs

1. **Select your sources**: Paste handbook URLs, blog roots, API docs, or sitemap links into the sidebar. The crawler respects domain limits and allowed path prefixes so you can focus on the sections that matter.
2. **Tune the chat persona**: Update the app title in the sidebar, e.g., “Kubernetes Operator SOP Assistant”, to match the content you ingest.
3. **Pick embeddings that fit your budget**: Use the bundled local sentence-transformer for zero API cost or switch to OpenAI’s `text-embedding-3-large` if you need higher accuracy. The index persists per session regardless of the embedding provider.
4. **Cache & reuse**: Download indexed data as a `.zip` (docs + vectors). Later, upload the pack to avoid re-crawling when you revisit the same sources.

---

## Quick Start (Local)
1. **Set up Python** (3.10+) and create a virtual environment
   ```bash
   uv venv {your_env_name}
   source {your_env_name}/bin/activate  # Windows: .venv\\Scripts\\activate
   uv pip install -r requirements.txt
   ```
2. **Run Streamlit**
   ```bash
   streamlit run rag_chatbot/app.py
   ```
3. **Configure the sidebar**
   - Choose the LLM backend: `browser-local` (default) or `openai`.
   - Provide your Ollama host/model or OpenAI API key as needed.
   - Select the emsbeddings backend (`local` vs `openai`).
   - Manage chat sessions: folder buttons switch threads, the pencil toggles inline rename (press Enter to save), and the trash icon removes a conversation.
   - For `browser-local`, the browser must reach the URL you provide (typically `http://localhost:11434`) and the endpoint has to return permissive CORS headers.
4. **Crawl or ingest**
   - **Crawl & Index**: give a list of seed URLs, optional path filters, and crawl limits.
   - **Load from Sitemap(s)**: paste sitemap URLs, set limits, and let the app fetch, chunk, and embed in bulk.
5. **Chat**
   - Ask questions in the bottom input; responses stream with inline citations.
   - Conversations retain context across turns until you delete the session.
6. **Persist / Reload**
   - `Download Index (.zip)` exports vectors + metadata.
   - `Upload Index (.zip)` restores an index without re-running the crawler.

> Need to validate your local model endpoint? Run `python rag_chatbot/test_ollama.py --host http://localhost:11434 --model llama3.1:8b`.

### Browser → Local Ollama flow

- When you pick the `browser-local` backend, the Python app still performs retrieval, but the final LLM call happens from the browser via a tiny component.
- Ollama must be running on the viewer's machine and expose CORS headers (`Access-Control-Allow-Origin: *`). If native Ollama lacks CORS, place a lightweight proxy (Node `local-cors-proxy`, Flask, nginx, etc.) in front of it.
- Because requests originate from the end user's browser, each viewer can keep their local models private while using a hosted Streamlit deployment.

---

## Notable UI Features
- **Adaptive Theme**: Launches in a bright, sky-blue palette, and you can switch to dark mode with Streamlit’s built-in settings (the custom theming respects both options).
- **Multi-chat Sidebar**: Quickly jump between different research threads, rename them inline, or delete when finished.
- **Dynamic Titles**: The app title and chat titles update automatically from your inputs or first user messages.
- **Local-First Retrieval**: Default embeddings run locally; switch to OpenAI only when you need managed accuracy.

---

## Technical Deep Dive

```
rag_chatbot/
├── app.py              # Streamlit UI, session management, crawling & chat orchestration
├── requirements.txt    # Python dependencies (Streamlit, Chroma, OpenAI, sentence-transformers, etc.)
├── .streamlit/
│   └── config.toml     # Dark theme configuration
├── rag/
│   ├── __init__.py
│   ├── crawl.py        # BFS crawler + sitemap loader tuned for documentation sites
│   ├── embeddings.py   # Local (sentence-transformers) and OpenAI embedding adapters
│   ├── index.py        # VectorIndex built on Chroma with FAISS/NumPy fallback
│   ├── llm.py          # OpenAI and Ollama chat clients with streaming + fallback logic
│   ├── persist.py      # Pack/unpack helpers for portable index snapshots
│   └── utils.py        # HTML cleaning and chunking utilities
└── test_ollama.py      # Standalone connectivity script for Ollama chat endpoints
```

### Architecture & Retrieval Pipeline
- **Local-first embeddings**: The default `sentence-transformers/all-MiniLM-L6-v2` model embeds chunks on your machine. This keeps experimentation free and private. Switching to OpenAI is a toggle away and the VectorIndex records which backend created the vectors to avoid dimension mismatches.
- **Chroma vector store**: The app writes chunks into a session-local `.rag_chroma/` directory. Chroma gives us efficient similarity search out of the box, while the code gracefully falls back to FAISS or NumPy if Chroma is unavailable.
- **Chunk persistence**: Downloaded `.zip` files include `docs.json`, `vecs.npy`, and metadata describing the embedding backend + model. Uploading restores the vectors immediately; if vectors are missing, the app re-embeds using your current settings.
- **Streaming LLM responses**: Ollama responses are streamed via Server-Sent Events with a non-streaming fallback to handle endpoints that buffer output. OpenAI chat completions use the `stream=True` API to keep the UI responsive.

### Customisation Hooks
- **Any documentation source**: Swap the seed URLs or sitemap list to target different sites. The crawler respects `allowed_path_prefixes` so you can scope to specific sections (e.g., `/docs/`, `/handbook/onboarding`).
- **Adaptive title**: The sidebar title field lets you rebrand the interface per project, and chat tabs auto-rename themselves to the first user utterance unless you override them.
- **Embeddings toggle**: Switch between `local` and `openai` embeddings without rebuilding the UI. The VectorIndex stores embed settings alongside vectors to ensure compatibility when reloading snapshots across machines.
- **LLM backend toggle**: Use the same interface for browser-mediated local experimentation and hosted inference (OpenAI). Only one environment variable—the OpenAI key—is required when switching to the cloud setup.

### Scaling & Operational Comfort
- **Batch-friendly ingestion**: Both the crawler and sitemap loader accept limits to throttle network usage. Chunk size/overlap sliders help balance recall vs. cost for different content densities.
- **Recoverable state**: All conversations, docs, and indices live in `st.session_state` so reruns preserve context until you intentionally clear it. Chat sessions are segregated, making it easy to maintain multiple research threads.
- **Night mode via config**: The `.streamlit/config.toml` theme is baked in, making it easy to deploy the dark UI across environments without extra flags.
- **Extensible adapters**: The modular structure (`rag/embeddings.py`, `rag/llm.py`) makes it straightforward to plug in additional embedding providers or LLM gateways if you scale beyond Ollama/OpenAI.

---

## Deployment Notes
- Set `app.py` as the Streamlit entry point and `requirements.txt` as the dependency manifest when deploying (e.g., Streamlit Community Cloud).
- Ensure the host running the app can reach your Ollama server if you rely on local models; otherwise, default to the OpenAI backend and provide an API key.
- For multi-user hosting, consider mounting a persistent volume for `.rag_chroma/` and enabling authentication around the UI.

## TODO
1. Support multiple backends other than OpenAI (e.g., Gemini, DeepSeek, etc.)
2. Robust support for concurrent access and multiple users at once.

Happy querying!
