from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

_COMPONENT_PATH = Path(__file__).parent / "frontend"

_browser_local_component = components.declare_component(
    "browser_local_ollama",
    path=str(_COMPONENT_PATH),
)


def invoke_browser_local(
    *,
    messages: List[Dict[str, str]],
    model: str,
    host: str,
    trigger: int,
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the browser-local Ollama component and return its response.

    Parameters
    ----------
    messages
        Chat messages to forward to the Ollama /api/chat endpoint.
    model
        Model identifier to send in the request payload.
    host
        Base URL (as seen by the browser) for the Ollama server.
    trigger
        Monotonic counter so the component can detect new invocations.
    key
        Optional Streamlit component key for state isolation.

    Returns
    -------
    Optional[Dict[str, Any]]
        Either the data sent from the component via ``Streamlit.setComponentValue``
        or ``None`` while the browser call is still pending.
    """

    return _browser_local_component(
        messages=messages,
        model=model,
        host=host,
        trigger=trigger,
        key=key,
    )

