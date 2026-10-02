"""Simple PDF RAG utilities for MarineWise AI."""

from __future__ import annotations

import io
import os
import re
from typing import Any

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


def get_embedder() -> SentenceTransformer:
    """Load the small local embedding model once per Streamlit process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Create simple character-based overlapping chunks."""
    text = _clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf(pdf_bytes: bytes, source_name: str) -> list[dict[str, Any]]:
    """Extract page text and page metadata from one PDF."""
    records: list[dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text("text"))
            if not text:
                continue
            for chunk_no, chunk in enumerate(_chunk_text(text), start=1):
                records.append(
                    {
                        "text": chunk,
                        "source": source_name,
                        "page": page_no,
                        "chunk": chunk_no,
                    }
                )
    return records


def build_index(pdf_items: list[tuple[str, bytes]], embedder: SentenceTransformer) -> dict[str, Any]:
    """Build a FAISS inner-product index using normalized embeddings."""
    records: list[dict[str, Any]] = []
    for name, data in pdf_items:
        records.extend(extract_pdf(data, name))

    if not records:
        raise ValueError("No readable text was found in the supplied PDF manuals.")

    embeddings = embedder.encode(
        [r["text"] for r in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return {
        "index": index,
        "records": records,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "manual_count": len(pdf_items),
    }


def search_index(rag_state: dict[str, Any], query: str, embedder: SentenceTransformer, k: int = 5) -> list[dict[str, Any]]:
    """Retrieve the most similar manual chunks."""
    if not rag_state or rag_state.get("index") is None:
        return []
    vector = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    k = min(k, len(rag_state["records"]))
    scores, indices = rag_state["index"].search(vector, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(rag_state["records"][int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def retrieve_context(results: list[dict[str, Any]], min_score: float = 0.28) -> tuple[str, list[dict[str, Any]]]:
    """Format only reasonably relevant chunks for the LLM."""
    relevant = [r for r in results if r["score"] >= min_score]
    context_parts = []
    for i, r in enumerate(relevant, start=1):
        context_parts.append(
            f"SOURCE {i}: {r['source']} | PAGE {r['page']} | similarity {r['score']:.3f}\n{r['text']}"
        )
    return "\n\n".join(context_parts), relevant


def download_google_drive_pdf(url: str) -> tuple[str, bytes]:
    """Download a publicly accessible Google Drive file as bytes.

    The link must point to a PDF that the app is allowed to access publicly.
    """
    match = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if not match:
        raise ValueError("Could not find a Google Drive file ID in that link.")
    file_id = match.group(1)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type and b"Google Drive" in response.content[:5000]:
        raise ValueError("Google Drive returned a sharing/confirmation page. Make the PDF accessible to anyone with the link and try again.")
    return f"Google Drive PDF {file_id}.pdf", response.content
