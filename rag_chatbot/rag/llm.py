from __future__ import annotations

import os

from typing import Iterable, List, Dict, Optional


class ChatModel:
    def stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        raise NotImplementedError


class OpenAIChat(ChatModel):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def stream(self, messages: List[Dict[str, str]]):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        for chunk in resp:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class OllamaChat(ChatModel):
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        import requests  # noqa: F401 - ensure available

        self.host = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.model = model or DEFAULT_OLLAMA_MODEL

    def stream(self, messages: List[Dict[str, str]]):
        import json
        import requests

        url = f"{self.host}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": True}
        yielded = False
        stream_error: Exception | None = None
        try:
            with requests.post(url, json=payload, stream=True, timeout=(5, 300)) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="ignore")
                    else:
                        line = raw_line
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        js = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if js.get("done"):
                        break
                    delta = js.get("message", {}).get("content", "")
                    if delta:
                        yielded = True
                        yield delta
        except requests.exceptions.RequestException as exc:
            stream_error = exc

        if yielded:
            return

        fallback_payload = {**payload, "stream": False}
        try:
            resp = requests.post(url, json=fallback_payload, timeout=(5, 300))
            resp.raise_for_status()
            data = resp.json()
            delta = data.get("message", {}).get("content", "")
            if delta:
                yield delta
                return
        except requests.exceptions.RequestException as exc:
            stream_error = exc

        if stream_error:
            raise RuntimeError(f"Ollama request failed: {stream_error}") from stream_error
        raise RuntimeError("Ollama returned no content.")


def get_chat_model(
    backend: str,
    openai_key: Optional[str] = None,
    ollama_host: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> ChatModel:
    if backend == "openai":
        if not openai_key:
            raise ValueError("OpenAI key required for openai backend")
        return OpenAIChat(api_key=openai_key)
    return OllamaChat(host=ollama_host, model=ollama_model)
