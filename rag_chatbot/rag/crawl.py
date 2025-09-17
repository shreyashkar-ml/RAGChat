from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import gzip
import io

from .utils import html_to_text


def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


def _allowed_path(url: str, allowed_prefixes: List[str]) -> bool:
    path = urlparse(url).path or "/"
    if not allowed_prefixes:
        return True
    return any(path.startswith(p.strip()) for p in allowed_prefixes if p.strip())


def _normalize(u: str) -> str:
    p = urlparse(u)
    # Drop fragments and query for indexing
    return p._replace(fragment="", query="").geturl()


def crawl_gitlab(
    start_urls: List[str],
    allowed_path_prefixes: List[str] | None = None,
    limit: int = 80,
    per_domain: int = 120,
) -> List[Dict]:
    """
    Lightweight BFS crawler for GitLab handbook/direction.
    Returns a list of {url, title, text} (page-level, prior to chunking).
    """

    allowed_path_prefixes = allowed_path_prefixes or ["/handbook", "/direction"]
    ua = {"User-Agent": "rag-chatbot/0.1"}

    q = deque()
    for s in start_urls:
        if s:
            q.append(s)

    visited = set()
    host_counts: Dict[str, int] = {}
    pages: List[Dict] = []

    while q and len(pages) < limit:
        url = q.popleft()
        url = _normalize(url)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=ua, timeout=20)
            if resp.status_code >= 400 or not resp.headers.get("content-type", "").startswith("text/html"):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.text.strip() if soup.title else url
            text = html_to_text(soup)
            pages.append({"url": url, "title": title, "text": text})

            # Enqueue links (same host, allowed paths)
            host = urlparse(url).netloc
            host_counts.setdefault(host, 0)
            host_counts[host] += 1
            if host_counts[host] > per_domain:
                continue

            for a in soup.find_all("a"):
                href = a.get("href") or ""
                if not href:
                    continue
                nxt = urljoin(url, href)
                if not nxt.startswith("http"):
                    continue
                if not _same_host(url, nxt):
                    continue
                if not _allowed_path(nxt, allowed_path_prefixes):
                    continue
                nxt = _normalize(nxt)
                if nxt not in visited:
                    q.append(nxt)
        except Exception:
            continue

    return pages


def _fetch_bytes(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    """Fetch raw bytes and content-type."""
    ua = {"User-Agent": "rag-chatbot/0.1"}
    r = requests.get(url, headers=ua, timeout=timeout)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "")


def _parse_sitemap_xml(data: bytes) -> Tuple[List[str], bool]:
    """Return (locs, is_index)."""
    try:
        root = ET.fromstring(data)
    except Exception:
        return [], False
    tag = root.tag.lower()
    locs: List[str] = []
    is_index = tag.endswith("sitemapindex")
    if is_index:
        for sm in root.findall("{*}sitemap/{*}loc"):
            if sm.text:
                locs.append(sm.text.strip())
    else:
        for url in root.findall("{*}url/{*}loc"):
            if url.text:
                locs.append(url.text.strip())
    return locs, is_index


def _load_sitemap(url: str) -> Tuple[List[str], bool]:
    data, ctype = _fetch_bytes(url)
    if url.endswith(".gz") or "gzip" in ctype:
        try:
            data = gzip.decompress(data)
        except Exception:
            # try via BytesIO
            try:
                data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
            except Exception:
                pass
    return _parse_sitemap_xml(data)


def sitemap_urls(sitemaps: List[str], limit: int | None = None) -> List[str]:
    """Collect URLs from sitemap or sitemap index URLs recursively (depth<=3)."""
    seen = set()
    out: List[str] = []
    q = deque([(u, 0) for u in sitemaps if u])
    while q:
        u, depth = q.popleft()
        try:
            locs, is_index = _load_sitemap(u)
        except Exception:
            continue
        for loc in locs:
            if is_index and depth < 3:
                if loc not in seen:
                    seen.add(loc)
                    q.append((loc, depth + 1))
            else:
                if loc not in seen:
                    seen.add(loc)
                    out.append(loc)
                    if limit and len(out) >= limit:
                        return out
    return out


def crawl_from_url_list(
    urls: List[str],
    allowed_path_prefixes: List[str] | None = None,
    limit: int | None = None,
    per_domain: int = 1_000_000,
) -> List[Dict]:
    """Fetch pages directly from a provided list of URLs, applying filters and limits."""
    allowed_path_prefixes = allowed_path_prefixes or []
    pages: List[Dict] = []
    host_counts: Dict[str, int] = {}
    for url in urls:
        if limit and len(pages) >= limit:
            break
        if not _allowed_path(url, allowed_path_prefixes):
            continue
        host = urlparse(url).netloc
        host_counts.setdefault(host, 0)
        if host_counts[host] >= per_domain:
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": "rag-chatbot/0.1"}, timeout=20)
            if resp.status_code >= 400 or not resp.headers.get("content-type", "").startswith("text/html"):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.text.strip() if soup.title else url
            text = html_to_text(soup)
            pages.append({"url": _normalize(url), "title": title, "text": text})
            host_counts[host] += 1
        except Exception:
            continue
    return pages
