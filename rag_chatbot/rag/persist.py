from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np


def pack_index(
    docs: List[Dict[str, Any]],
    vecs: np.ndarray | None,
    meta: Dict[str, Any],
) -> bytes:
    """Pack docs + vectors + meta into a single zip blob."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # docs.json
        z.writestr("docs.json", json.dumps(docs, ensure_ascii=False).encode("utf-8"))
        # vecs.npy (optional)
        if vecs is not None:
            vbuf = io.BytesIO()
            np.save(vbuf, vecs.astype("float32"))
            z.writestr("vecs.npy", vbuf.getvalue())
            meta = {**meta, "dim": int(vecs.shape[1])}
        # meta.json
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    return buf.getvalue()


def unpack_index(blob: bytes) -> Tuple[List[Dict[str, Any]], np.ndarray | None, Dict[str, Any]]:
    """Unpack docs + vectors + meta from a zip blob."""
    with zipfile.ZipFile(io.BytesIO(blob), mode="r") as z:
        with z.open("docs.json") as f:
            docs = json.loads(f.read().decode("utf-8"))
        vecs = None
        if "vecs.npy" in z.namelist():
            with z.open("vecs.npy") as f:
                b = f.read()
                buf = io.BytesIO(b)
                vecs = np.load(buf)
        with z.open("meta.json") as f:
            meta = json.loads(f.read().decode("utf-8"))
    return docs, vecs, meta

