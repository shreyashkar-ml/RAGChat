from __future__ import annotations

from typing import List

from bs4 import BeautifulSoup


def html_to_text(soup: BeautifulSoup) -> str:
    # Remove noisy elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    if chunk_size <= 0:
        return [text]
    parts: List[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - max(0, overlap))
    while i < n:
        parts.append(text[i : i + chunk_size])
        i += step
    return parts

