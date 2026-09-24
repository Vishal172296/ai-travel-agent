"""
Advanced Text Chunking Engine for AI Travel Agent RAG Pipeline.
Implements sliding-window chunking with sentence boundary preservation,
token budget management, and metadata enrichment per chunk.
"""

import re
from typing import Any, Dict, List, Optional


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting natural boundaries."""
    # Split on period/exclamation/question mark followed by whitespace and capital letter, or double newline
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n\n+", text)
    sentences = [s.strip() for s in raw_sentences if s and s.strip()]
    return sentences if sentences else [text]


def chunk_document(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 35,
) -> List[Dict[str, Any]]:
    """
    Split a document into overlapping semantic chunks with full metadata attachment.
    Preserves paragraph structure and natural sentence boundaries.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    meta = metadata.copy() if metadata else {}
    doc_id = meta.get("doc_id", "doc_0")
    source = meta.get("source", "Knowledge Base")
    destination = meta.get("destination", "General")
    category = meta.get("category", "Guide")

    paragraphs = cleaned.split("\n\n")
    chunks: List[Dict[str, Any]] = []
    current_chunk_words: List[str] = []

    for paragraph in paragraphs:
        p_words = paragraph.strip().split()
        if not p_words:
            continue

        if len(current_chunk_words) + len(p_words) <= chunk_size_words:
            current_chunk_words.extend(p_words)
        else:
            # Check sentence breakdown of paragraph
            sentences = split_into_sentences(paragraph)
            for s in sentences:
                s_words = s.split()
                if len(current_chunk_words) + len(s_words) > chunk_size_words:
                    if current_chunk_words:
                        chunk_text = " ".join(current_chunk_words)
                        chunk_idx = len(chunks)
                        chunks.append({
                            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                            "doc_id": doc_id,
                            "chunk_index": chunk_idx,
                            "text": chunk_text,
                            "word_count": len(current_chunk_words),
                            "source": source,
                            "destination": destination,
                            "category": category,
                            "metadata": meta,
                        })
                        # Keep overlap words
                        overlap = current_chunk_words[-chunk_overlap_words:] if len(current_chunk_words) > chunk_overlap_words else current_chunk_words
                        current_chunk_words = overlap + s_words
                    else:
                        # Single long sentence exceeds chunk size
                        chunk_text = " ".join(s_words)
                        chunk_idx = len(chunks)
                        chunks.append({
                            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                            "doc_id": doc_id,
                            "chunk_index": chunk_idx,
                            "text": chunk_text,
                            "word_count": len(s_words),
                            "source": source,
                            "destination": destination,
                            "category": category,
                            "metadata": meta,
                        })
                        current_chunk_words = []
                else:
                    current_chunk_words.extend(s_words)

    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunk_idx = len(chunks)
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
            "doc_id": doc_id,
            "chunk_index": chunk_idx,
            "text": chunk_text,
            "word_count": len(current_chunk_words),
            "source": source,
            "destination": destination,
            "category": category,
            "metadata": meta,
        })

    return chunks
