import json
import sys
from pathlib import Path
from typing import List, Dict

import streamlit as st

# Ensure project root is importable when launched via `streamlit run`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_chatbot.rag.embeddings import get_embedder
from rag_chatbot.rag.index import VectorIndex
from rag_chatbot.rag.crawl import crawl_gitlab, sitemap_urls, crawl_from_url_list
from rag_chatbot.rag.llm import get_chat_model
from rag_chatbot.rag.utils import chunk_text
from rag_chatbot.rag.persist import pack_index, unpack_index

DEFAULT_TITLE = "GitLab Handbook and Directions Expert Querying"

st.set_page_config(page_title=DEFAULT_TITLE, layout="wide")

THEMES = {
    "light": {
        "background": "#F7F9FC",
        "sidebar_bg": "#FFFFFF",
        "text": "#111827",
        "card_bg": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "border": "#E2E8F0",
        "code_bg": "#F1F4FB",
        "button_bg": "#38B6FF",
        "button_hover": "#1A94FF",
        "button_text": "#0B1220",
        "button_border": "#1A94FF",
        "link": "#1A94FF",
        "shadow": "0 2px 6px rgba(15, 23, 42, 0.08)",
        "status_border": "#E2E8F0",
    },
    "dark": {
        "background": "#0E1117",
        "sidebar_bg": "#1C1F26",
        "text": "#F5F6F7",
        "card_bg": "#141821",
        "input_bg": "#1C2430",
        "border": "#2A2F3A",
        "code_bg": "#1C2430",
        "button_bg": "#38B6FF",
        "button_hover": "#1A94FF",
        "button_text": "#0B1220",
        "button_border": "#1A94FF",
        "link": "#38B6FF",
        "shadow": "0 2px 10px rgba(4, 9, 20, 0.6)",
        "status_border": "#1F2A38",
    },
}


def _build_theme_block(theme_name: str, colors: Dict[str, str]) -> str:
    prefix = f"body[data-baseweb='{theme_name}']"
    return f"""
    {prefix} {{
        background-color: {colors['background']} !important;
        color: {colors['text']} !important;
    }}
    {prefix} .stApp {{
        background-color: {colors['background']} !important;
        color: {colors['text']} !important;
    }}
    {prefix} #root,
    {prefix} .main,
    {prefix} [data-testid="stAppViewContainer"],
    {prefix} section.main,
    {prefix} section.main > div,
    {prefix} div[data-testid="stVerticalBlock"] {{
        background-color: {colors['background']} !important;
    }}
    {prefix} .main .block-container {{
        background-color: {colors['background']} !important;
        padding-top: 1.2rem !important;
        padding-bottom: 0.6rem !important;
        margin-bottom: 0 !important;
    }}
    {prefix} header[data-testid="stHeader"],
    {prefix} footer {{
        background-color: {colors['background']} !important;
    }}
    {prefix} footer {{
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }}
    {prefix} [data-testid="stSidebar"] {{
        background-color: {colors['sidebar_bg']} !important;
        color: {colors['text']} !important;
    }}
    {prefix} .stButton > button {{
        background-color: {colors['button_bg']} !important;
        color: {colors['button_text']} !important;
        border: 1px solid {colors['button_border']} !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }}
    {prefix} .stButton > button:hover {{
        background-color: {colors['button_hover']} !important;
        border: 1px solid {colors['button_hover']} !important;
    }}
    {prefix} .stTextInput > div > div > input,
    {prefix} .stTextArea textarea,
    {prefix} .stSelectbox > div > div,
    {prefix} .stSlider > div {{
        background-color: {colors['input_bg']} !important;
        color: {colors['text']} !important;
        border-radius: 6px !important;
        border: 1px solid {colors['border']} !important;
    }}
    {prefix} .stTextInput > div > div > input:focus,
    {prefix} .stTextArea textarea:focus,
    {prefix} .stSelectbox > div > div:focus {{
        border: 1px solid {colors['button_border']} !important;
    }}
    {prefix} label,
    {prefix} .stMarkdown,
    {prefix} .stMarkdown p,
    {prefix} .stMarkdown span,
    {prefix} .stRadio,
    {prefix} .stSelectbox label,
    {prefix} .stSlider label {{
        color: {colors['text']} !important;
    }}
    {prefix} a {{
        color: {colors['link']} !important;
    }}
    {prefix} div[data-testid="stChatMessage"] {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text']} !important;
        border-radius: 12px !important;
        box-shadow: {colors['shadow']};
    }}
    {prefix} div[data-testid="stChatMessage"] pre,
    {prefix} div[data-testid="stChatMessage"] code {{
        background-color: {colors['code_bg']} !important;
        color: {colors['text']} !important;
    }}
    {prefix} div[data-testid="stChatInput"] {{
        background-color: {colors['background']} !important;
        padding: 0.4rem 0 0.4rem 0 !important;
        margin: 0 !important;
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
    }}
    {prefix} div[data-testid="stChatInput"] > div {{
        background-color: {colors['card_bg']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 10px !important;
        margin-bottom: 0 !important;
    }}
    {prefix} div[data-testid="stChatInput"] textarea {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text']} !important;
    }}
    {prefix} div[data-testid="stChatInput"] textarea:focus {{
        border: none !important;
    }}
    {prefix} div[class*="status-container"] {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text']} !important;
        border-radius: 10px !important;
        border: 1px solid {colors['status_border']} !important;
    }}
    """


def _apply_theme():
    css_blocks = [_build_theme_block(name, colors) for name, colors in THEMES.items()]
    st.markdown(
        "<style>" + "\n".join(css_blocks) + "</style>",
        unsafe_allow_html=True,
    )


_apply_theme()


def _finalize_chat_rename(chat_id: str, state_key: str, new_title: str | None = None):
    chats = st.session_state.chats
    if chat_id not in chats:
        if state_key in st.session_state:
            del st.session_state[state_key]
        st.session_state.pop("_chat_rename_queue", None)
        return

    original_title = chats[chat_id]["title"]
    candidate = (new_title if new_title is not None else st.session_state.get(state_key, "")).strip()
    chats[chat_id]["title"] = candidate or original_title
    st.session_state.chat_rename_target = None
    if state_key in st.session_state:
        del st.session_state[state_key]
    st.session_state.pop("_chat_rename_queue", None)


def _queue_chat_rename(chat_id: str, state_key: str):
    st.session_state["_chat_rename_queue"] = (
        chat_id,
        state_key,
        st.session_state.get(state_key, ""),
    )


def _cancel_chat_rename(state_key: str):
    if state_key in st.session_state:
        del st.session_state[state_key]
    st.session_state.chat_rename_target = None
    st.session_state.pop("_chat_rename_queue", None)


# Session State
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_rename_target" not in st.session_state:
    st.session_state.chat_rename_target = None

pending_rename = st.session_state.pop("_chat_rename_queue", None)
if pending_rename:
    if isinstance(pending_rename, tuple) and len(pending_rename) == 3:
        queued_chat_id, queued_state_key, queued_value = pending_rename
        _finalize_chat_rename(queued_chat_id, queued_state_key, new_title=queued_value)
    else:
        st.session_state.pop("_chat_rename_queue", None)


def _delete_chat(chat_id: str):
    if chat_id not in st.session_state.chats:
        return
    # Remove any rename state tied to this chat
    rename_key = f"chat_title_input_{chat_id}"
    if rename_key in st.session_state:
        del st.session_state[rename_key]
    del st.session_state.chats[chat_id]
    if st.session_state.chat_rename_target == chat_id:
        st.session_state.chat_rename_target = None
    if st.session_state.current_chat_id == chat_id:
        remaining = list(st.session_state.chats.keys())
        if remaining:
            st.session_state.current_chat_id = remaining[-1]
        else:
            fresh = _create_chat()
            st.session_state.current_chat_id = fresh
    st.session_state.messages = _get_current_chat()["messages"]


def _create_chat(title: str | None = None) -> str:
    st.session_state.chat_counter += 1
    chat_id = f"chat-{st.session_state.chat_counter}"
    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": title or f"Chat {st.session_state.chat_counter}",
        "messages": [],
    }
    return chat_id


def _ensure_current_chat() -> str:
    cid = st.session_state.current_chat_id
    if not cid or cid not in st.session_state.chats:
        cid = _create_chat()
        st.session_state.current_chat_id = cid
    return cid


def _get_current_chat():
    cid = _ensure_current_chat()
    return st.session_state.chats[cid]


def _switch_chat(chat_id: str):
    if chat_id not in st.session_state.chats:
        return
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chats[chat_id]["messages"]


current_chat = _get_current_chat()
st.session_state.messages = current_chat["messages"]  # backward compatibility

if "docs" not in st.session_state:
    st.session_state.docs = []      # [{id, title, url, text}]
if "index" not in st.session_state:
    st.session_state.index = None   # VectorIndex
if "index_info" not in st.session_state:
    st.session_state.index_info = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "llm_backend": "ollama",          # "openai" or "ollama"
        "openai_api_key": "",
        "ollama_host": "http://localhost:11434",
        "ollama_model": "",
        "embed_backend": "local",         # "openai" or "local"
        "openai_embed_model": "text-embedding-3-large",
        "local_embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "app_title": DEFAULT_TITLE,
    }


def ensure_index() -> VectorIndex | None:
    if st.session_state.index is None and st.session_state.docs:
        embed_settings = {
            "embed_backend": st.session_state.settings["embed_backend"],
            "openai_embed_model": st.session_state.settings.get("openai_embed_model"),
            "local_embed_model": st.session_state.settings.get("local_embed_model"),
        }
        embedder = get_embedder(
            backend=embed_settings["embed_backend"],
            openai_key=st.session_state.settings.get("openai_api_key") or None,
            openai_model=embed_settings.get("openai_embed_model"),
            local_model=embed_settings.get("local_embed_model"),
        )
        idx = VectorIndex(embedder=embedder, persist_path=PROJECT_ROOT / ".rag_chroma")
        idx.add_documents(st.session_state.docs)
        st.session_state.index = idx
        st.session_state.index_info = embed_settings
    return st.session_state.index


def add_message(role: str, content: str):
    chat = _get_current_chat()
    chat["messages"].append({"role": role, "content": content})
    st.session_state.messages = chat["messages"]
    if role == "user" and chat["title"].startswith("Chat "):
        first_line = content.strip().splitlines()[0] if content.strip() else "New chat"
        if len(first_line) > 45:
            first_line = first_line[:42].rstrip() + "…"
        chat["title"] = first_line or chat["title"]


# Sidebar
with st.sidebar:
    st.header("Settings")
    app_title_input = st.text_input(
        "App title",
        value=st.session_state.settings.get("app_title", DEFAULT_TITLE),
        help="Customize the main heading shown in the app body.",
    )
    st.session_state.settings["app_title"] = app_title_input.strip() or DEFAULT_TITLE

    st.session_state.settings["llm_backend"] = st.selectbox(
        "LLM backend", ["openai", "ollama"], index=1
    )
    if st.session_state.settings["llm_backend"] == "openai":
        st.session_state.settings["openai_api_key"] = st.text_input(
            "OpenAI API Key", type="password"
        )
    else:
        st.session_state.settings["ollama_host"] = st.text_input(
            "Ollama Host", value=st.session_state.settings["ollama_host"]
        )
        st.session_state.settings["ollama_model"] = st.text_input(
            "Ollama Model",
            value=st.session_state.settings.get("ollama_model", ""),
            placeholder="Enter the Ollama model name (e.g. llama3.1:8b-instruct)",
        )

    embed_options = ["local", "openai"]
    embed_default = (
        embed_options.index(st.session_state.settings["embed_backend"])
        if st.session_state.settings["embed_backend"] in embed_options
        else 0
    )
    st.session_state.settings["embed_backend"] = st.selectbox(
        "Embeddings backend", embed_options, index=embed_default
    )
    if st.session_state.settings["embed_backend"] == "openai":
        st.session_state.settings["openai_embed_model"] = st.text_input(
            "OpenAI embed model", value=st.session_state.settings["openai_embed_model"]
        )
    else:
        st.session_state.settings["local_embed_model"] = st.text_input(
            "Local embed model", value=st.session_state.settings["local_embed_model"]
        )

    st.divider()
    st.subheader("Chat Sessions")
    if st.button("➕ New chat", use_container_width=True):
        new_chat_id = _create_chat()
        _switch_chat(new_chat_id)
        st.session_state.chat_rename_target = new_chat_id
    chat_ids = list(st.session_state.chats.keys())
    current_chat_id = st.session_state.current_chat_id
    for cid in chat_ids:
        chat_title = st.session_state.chats[cid]["title"]
        cols = st.columns([0.68, 0.16, 0.16])
        rename_key = f"chat_title_input_{cid}"
        is_renaming = st.session_state.chat_rename_target == cid

        if is_renaming:
            st.session_state.setdefault(rename_key, chat_title)
            cols[0].text_input(
                "Rename chat",
                key=rename_key,
                label_visibility="collapsed",
                on_change=_queue_chat_rename,
                args=(cid, rename_key),
            )
            if cols[1].button("✅", key=f"chat_rename_confirm_{cid}", help="Save chat title"):
                _finalize_chat_rename(cid, rename_key, new_title=st.session_state.get(rename_key, ""))
            if cols[2].button("✖️", key=f"chat_rename_cancel_{cid}", help="Cancel rename"):
                _cancel_chat_rename(rename_key)
        else:
            select_kwargs = {
                "use_container_width": True,
                "key": f"chat_select_btn_{cid}",
            }
            if cid == current_chat_id:
                select_kwargs["type"] = "primary"
            chat_label = st.session_state.chats[cid]["title"]
            if cols[0].button(chat_label, **select_kwargs):
                _switch_chat(cid)
                current_chat_id = cid
                st.session_state.chat_rename_target = None

            if cols[1].button("✏️", key=f"chat_rename_btn_{cid}", help="Rename this chat"):
                st.session_state.chat_rename_target = cid
                st.session_state[rename_key] = chat_title

            if cols[2].button("🗑️", key=f"chat_delete_btn_{cid}", help="Delete this chat"):
                _delete_chat(cid)
                st.rerun()
    st.divider()
    st.subheader("Dataset: GitLab Docs")
    start_urls = st.text_area(
        "Start URLs (one per line)",
        value="\n".join([
            "https://handbook.gitlab.com/",
            "https://about.gitlab.com/direction/",
        ]),
        height=90,
    ).strip().splitlines()
    max_pages = st.slider("Max pages to crawl", 10, 400, 80, 10)
    max_per_domain = st.slider("Max per domain", 10, 400, 120, 10)
    allowed_paths = st.text_input(
        "Allowed path prefixes (comma)", value="/handbook,/direction,/"
    ).split(",")
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 10)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Crawl & Index", type="primary"):
            with st.status("Crawling GitLab docs…", expanded=True) as status:
                try:
                    docs_raw = crawl_gitlab(
                        start_urls=start_urls,
                        allowed_path_prefixes=[p.strip() for p in allowed_paths if p.strip()],
                        limit=max_pages,
                        per_domain=max_per_domain,
                    )
                    st.write(f"Fetched {len(docs_raw)} pages. Chunking…")
                    docs: List[Dict] = []
                    did = 0
                    for d in docs_raw:
                        chunks = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
                        for ch in chunks:
                            docs.append({
                                "id": f"{d['url']}#chunk-{did}",
                                "title": d.get("title") or d["url"],
                                "url": d["url"],
                                "text": ch,
                            })
                            did += 1
                    st.session_state.docs = docs
                    st.write(f"Prepared {len(docs)} chunks. Building index…")
                    st.session_state.index = None
                    st.session_state.index_info = None
                    ensure_index()
                    status.update(label="Index ready", state="complete")
                except Exception as e:
                    status.update(label=f"Failed: {e}", state="error")
    with col2:
        if st.button("Clear Index"):
            st.session_state.index = None
            st.session_state.docs = []
            st.session_state.index_info = None

    index_ready = st.session_state.index is not None
    embed_info = st.session_state.index_info["embed_backend"] if (st.session_state.index_info and st.session_state.index_info.get("embed_backend")) else st.session_state.settings["embed_backend"]
    st.caption(
        f"Docs: {len(st.session_state.docs)} · Index: {'ready' if index_ready else 'not built'} · Embeddings: {embed_info}"
    )

    # Sitemaps ingestion
    st.subheader("Sitemaps (Full Coverage)")
    sitemap_list = st.text_area(
        "Sitemap URLs (one per line)",
        value="\n".join([
            "https://about.gitlab.com/sitemap.xml",
        ]),
        height=80,
    ).strip().splitlines()
    max_urls = st.slider("Max URLs from sitemaps", 50, 10000, 1000, 50)
    if st.button("Load from Sitemap(s) & Index", type="secondary"):
        with st.status("Loading sitemap(s)…", expanded=True) as status:
            try:
                urls = sitemap_urls(sitemap_list, limit=max_urls)
                st.write(f"Found {len(urls)} URLs in sitemaps. Filtering & fetching…")
                pages = crawl_from_url_list(
                    urls,
                    allowed_path_prefixes=[p.strip() for p in allowed_paths if p.strip()],
                    limit=max_pages,
                    per_domain=max_per_domain,
                )
                st.write(f"Fetched {len(pages)} pages. Chunking…")
                docs: List[Dict] = []
                did = 0
                for d in pages:
                    chunks = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
                    for ch in chunks:
                        docs.append({
                            "id": f"{d['url']}#chunk-{did}",
                            "title": d.get("title") or d["url"],
                            "url": d["url"],
                            "text": ch,
                        })
                        did += 1
                st.session_state.docs = docs
                st.write(f"Prepared {len(docs)} chunks. Building index…")
                st.session_state.index = None
                st.session_state.index_info = None
                ensure_index()
                status.update(label="Index ready from sitemaps", state="complete")
            except Exception as e:
                status.update(label=f"Sitemap ingest failed: {e}", state="error")

    # Persistence (Download / Upload)
    st.subheader("Persistence")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        # Prepare a downloadable index pack
        idx = st.session_state.index
        docs = st.session_state.docs
        if docs and idx:
            docs_out, vecs_out = idx.dump()
            info = st.session_state.index_info or {}
            meta = {
                "embed_backend": info.get("embed_backend") or st.session_state.settings["embed_backend"],
                "openai_embed_model": info.get("openai_embed_model") or st.session_state.settings.get("openai_embed_model"),
                "local_embed_model": info.get("local_embed_model") or st.session_state.settings.get("local_embed_model"),
                "source": "gitlab",
            }
            blob = pack_index(docs_out, vecs_out, meta)
            st.download_button(
                "Download Index (.zip)",
                data=blob,
                file_name="gitlab_index.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.button("Download Index (.zip)", disabled=True, use_container_width=True)

    with pcol2:
        uploaded = st.file_uploader("Upload Index (.zip)", type=["zip"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                docs_in, vecs_in, meta_in = unpack_index(uploaded.read())
                # Load directly into index without re-embedding
                # Create a VectorIndex with a placeholder embedder; we'll rely on current settings
                # Try to create an embedder; if it fails (e.g., missing OpenAI key), load index anyway
                embedder = None
                try:
                    embedder = get_embedder(
                        backend=st.session_state.settings["embed_backend"],
                        openai_key=st.session_state.settings.get("openai_api_key") or None,
                        openai_model=st.session_state.settings.get("openai_embed_model"),
                        local_model=st.session_state.settings.get("local_embed_model"),
                    )
                except Exception as _e:
                    embedder = None
                new_index = VectorIndex(embedder=embedder, persist_path=PROJECT_ROOT / ".rag_chroma")
                if vecs_in is None:
                    st.warning("Uploaded pack lacks vectors; will re-embed on first search.")
                    # store docs; vectors will be built by ensure_index (re-embedding)
                    st.session_state.docs = docs_in
                    st.session_state.index = None
                    st.session_state.index_info = None
                else:
                    new_index.load(docs_in, vecs_in)
                    st.session_state.docs = docs_in
                    st.session_state.index = new_index
                    st.session_state.index_info = {
                        "embed_backend": meta_in.get("embed_backend") if meta_in else st.session_state.settings["embed_backend"],
                        "openai_embed_model": meta_in.get("openai_embed_model") if meta_in else st.session_state.settings.get("openai_embed_model"),
                        "local_embed_model": meta_in.get("local_embed_model") if meta_in else st.session_state.settings.get("local_embed_model"),
                    }
                # Optionally align settings to meta for dimension compatibility
                if meta_in:
                    be = meta_in.get("embed_backend")
                    if be and be in ("openai", "local"):
                        st.session_state.settings["embed_backend"] = be
                        if be == "openai" and meta_in.get("openai_embed_model"):
                            st.session_state.settings["openai_embed_model"] = meta_in["openai_embed_model"]
                        if be == "local" and meta_in.get("local_embed_model"):
                            st.session_state.settings["local_embed_model"] = meta_in["local_embed_model"]
                if st.session_state.index_info is None and st.session_state.index is not None:
                    st.session_state.index_info = {
                        "embed_backend": st.session_state.settings["embed_backend"],
                        "openai_embed_model": st.session_state.settings.get("openai_embed_model"),
                        "local_embed_model": st.session_state.settings.get("local_embed_model"),
                    }
                if embedder is None:
                    st.info("Index loaded. Provide embeddings backend credentials before searching.")
                else:
                    st.success("Index loaded from upload.")
            except Exception as e:
                st.error(f"Failed to load index: {e}")

app_title = st.session_state.settings.get("app_title", DEFAULT_TITLE) or DEFAULT_TITLE
st.title(app_title)
st.caption("Set up your preferred LLM in the sidebar, crawl once, and start asking questions.")

if st.button("Clear Current Chat"):
    chat = _get_current_chat()
    chat["messages"].clear()
    st.session_state.messages = chat["messages"]

# Render chat messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Ask about GitLab docs…")
if prompt:
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check index
    index = ensure_index()
    if not index:
        with st.chat_message("assistant"):
            st.warning("Please crawl & index the docs first from the sidebar.")
    else:
        # Ensure embedder is configured for searching
        if getattr(index, "embedder", None) is None:
            info = st.session_state.index_info or {
                "embed_backend": st.session_state.settings["embed_backend"],
                "openai_embed_model": st.session_state.settings.get("openai_embed_model"),
                "local_embed_model": st.session_state.settings.get("local_embed_model"),
            }
            try:
                index.embedder = get_embedder(
                    backend=info["embed_backend"],
                    openai_key=st.session_state.settings.get("openai_api_key") or None,
                    openai_model=info.get("openai_embed_model"),
                    local_model=info.get("local_embed_model"),
                )
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Embeddings not configured: {e}")
                st.stop()

        # Retrieve top docs
        try:
            hits = index.search(prompt, top_k=6)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(
                    "Search failed. Ensure your embeddings backend matches the index "
                    f"(try switching to {st.session_state.settings['embed_backend']} and setting credentials).\n\n{e}"
                )
            st.stop()
        context_blocks = []
        citations = []
        for i, h in enumerate(hits, 1):
            context_blocks.append(f"[{i}] {h['title']} — {h['url']}\n{h['text'][:600]}")
            citations.append((i, h['title'], h['url']))
        context = "\n\n".join(context_blocks)

        # Build system/user messages
        sys_prompt = (
            "You are a precise assistant for GitLab handbook and direction. "
            "Cite sources like [1], [2] where relevant."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"},
        ]

        # Get LLM
        chat = get_chat_model(
            backend=st.session_state.settings["llm_backend"],
            openai_key=st.session_state.settings.get("openai_api_key") or None,
            ollama_host=st.session_state.settings.get("ollama_host"),
            ollama_model=st.session_state.settings.get("ollama_model"),
        )

        # Stream tokens
        with st.chat_message("assistant"):
            container = st.empty()
            buf = ""
            try:
                for tok in chat.stream(messages):
                    if tok:
                        buf += tok
                        container.markdown(buf)
                # Append citations block
                if citations:
                    cites_md = "\n\n" + "\n".join(
                        [f"[{i}] {title} — {url}" for i, title, url in citations]
                    )
                    buf += cites_md
                    container.markdown(buf)
                add_message("assistant", buf)
            except Exception as e:
                container.error(f"Generation failed: {e}")
