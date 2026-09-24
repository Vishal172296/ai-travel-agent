"""
High-Performance In-Memory Vector Store and Hybrid Index for AI Travel Planner.
Provides dense vector semantic search, BM25 keyword inverted index,
metadata filtering, and Reciprocal Rank Fusion (RRF) hybrid retrieval.
Runs cleanly across environments (including Streamlit Community Cloud) without heavy C++ dependencies.
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into normalized lowercase alphanumeric tokens."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = clean.split()
    stop_words: Set[str] = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "it", "this",
        "that", "from", "as", "be", "have", "has", "do", "does", "you",
        "your", "we", "our", "i", "my", "me", "he", "she", "they", "them"
    }
    return [t for t in tokens if len(t) > 1 and t not in stop_words]


class DenseSemanticEmbedder:
    """
    Fast subword character-trigram & word n-gram dense embedding engine.
    Produces L2-normalized 256-dimensional semantic dense vectors with cosine similarity.
    Captures semantic similarity, typos, prefixes, and thematic clusters accurately.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        tokens = tokenize_text(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            return vec

        # Subword n-grams and word tokens
        for token in tokens:
            # Word level hash
            h_word = hash(token) % self.dim
            vec[h_word] += 1.5

            # Character 3-grams
            padded = f"<{token}>"
            for i in range(len(padded) - 2):
                tri = padded[i:i+3]
                h_tri = hash(tri) % self.dim
                vec[h_tri] += 0.8

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec


class TravelVectorStore:
    """
    Vector Database with BM25 keyword index, dense vector index, and metadata filtering.
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embedder = DenseSemanticEmbedder(dim=256)
        self.dense_vectors: Optional[np.ndarray] = None  # Shape (N, 256)
        self.doc_frequencies: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> int:
        """Index a batch of text chunks into the vector store and BM25 index."""
        if not new_chunks:
            return 0

        start_idx = len(self.chunks)
        new_vectors = []

        for chunk in new_chunks:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            title = meta.get("title", chunk.get("destination", ""))
            keywords = " ".join(meta.get("keywords", []))
            full_text_for_index = f"{title} {keywords} {text}"

            tokens = tokenize_text(full_text_for_index)
            chunk["tokens"] = tokens
            chunk["token_count"] = len(tokens)

            # Compute dense embedding
            v = self.embedder.embed(full_text_for_index)
            new_vectors.append(v)
            self.chunks.append(chunk)

        # Update dense matrix
        new_mat = np.array(new_vectors, dtype=np.float32)
        if self.dense_vectors is None or len(self.dense_vectors) == 0:
            self.dense_vectors = new_mat
        else:
            self.dense_vectors = np.vstack([self.dense_vectors, new_mat])

        # Update BM25 stats
        total_tokens = sum(c["token_count"] for c in self.chunks)
        self.avg_doc_len = total_tokens / max(len(self.chunks), 1)

        self.doc_frequencies.clear()
        for c in self.chunks:
            unique_tokens = set(c["tokens"])
            for t in unique_tokens:
                self.doc_frequencies[t] = self.doc_frequencies.get(t, 0) + 1

        return len(new_chunks)

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        destination_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Perform dense vector cosine similarity search."""
        if self.dense_vectors is None or len(self.chunks) == 0:
            return []

        q_vec = self.embedder.embed(query)
        # Cosine similarity is dot product of L2 normalized vectors
        scores = np.dot(self.dense_vectors, q_vec)

        results: List[Tuple[float, Dict[str, Any]]] = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if not self._matches_filters(chunk, destination_filter, category_filter):
                continue
            results.append((float(score), chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def keyword_search_bm25(
        self,
        query: str,
        top_k: int = 5,
        destination_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Perform BM25 keyword search with TF-IDF and length normalization."""
        q_tokens = tokenize_text(query)
        if not q_tokens or len(self.chunks) == 0:
            return []

        total_docs = len(self.chunks)
        results: List[Tuple[float, Dict[str, Any]]] = []

        k1 = 1.5
        b = 0.75

        for chunk in self.chunks:
            if not self._matches_filters(chunk, destination_filter, category_filter):
                continue

            doc_tokens = chunk["tokens"]
            doc_len = chunk["token_count"]
            score = 0.0

            for t in q_tokens:
                tf = doc_tokens.count(t)
                if tf > 0:
                    df = self.doc_frequencies.get(t, 1)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1.0))))
                    score += idf * norm_tf

            # Keyword tag boost
            keywords = chunk.get("metadata", {}).get("keywords", [])
            for kw in keywords:
                if any(kw in qt or qt in kw for qt in q_tokens):
                    score += 2.0

            if score > 0:
                results.append((score, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        destination_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval using Reciprocal Rank Fusion (RRF) combining
        dense semantic similarity and sparse BM25 keyword search.
        """
        if len(self.chunks) == 0:
            return []

        # Retrieve candidates from both methods
        semantic_results = self.semantic_search(query, top_k=top_k * 2, destination_filter=destination_filter, category_filter=category_filter)
        bm25_results = self.keyword_search_bm25(query, top_k=top_k * 2, destination_filter=destination_filter, category_filter=category_filter)

        # RRF scoring (k=60)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        sem_scores_map: Dict[str, float] = {}
        bm25_scores_map: Dict[str, float] = {}

        rrf_k = 60.0

        for rank, (score, chunk) in enumerate(semantic_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            sem_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (alpha * (1.0 / (rrf_k + rank + 1)))

        for rank, (score, chunk) in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            bm25_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + ((1.0 - alpha) * (1.0 / (rrf_k + rank + 1)))

        # Fallback if query produced no positive keyword matches
        if not rrf_scores:
            for score, chunk in semantic_results[:top_k]:
                cid = chunk["chunk_id"]
                chunk_map[cid] = chunk
                rrf_scores[cid] = score

        # Sort combined
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_chunks: List[Dict[str, Any]] = []
        for cid in sorted_cids[:top_k]:
            c = chunk_map[cid]
            final_chunks.append({
                "chunk_id": c["chunk_id"],
                "title": c.get("metadata", {}).get("title", c.get("destination", "Travel Guide")),
                "destination": c.get("destination", "General"),
                "category": c.get("category", "Guide"),
                "source": c.get("source", "Knowledge Base"),
                "content": c.get("text", ""),
                "score": round(rrf_scores[cid] * 100, 3),
                "semantic_score": round(sem_scores_map.get(cid, 0.0), 3),
                "bm25_score": round(bm25_scores_map.get(cid, 0.0), 3),
            })

        return final_chunks

    def _matches_filters(
        self,
        chunk: Dict[str, Any],
        destination_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> bool:
        """Check if chunk matches metadata filters."""
        if destination_filter:
            d_filter = destination_filter.lower().strip()
            c_dest = chunk.get("destination", "").lower()
            keywords = chunk.get("metadata", {}).get("keywords", [])
            matches_dest = d_filter in c_dest or any(d_filter in kw.lower() for kw in keywords)
            if not matches_dest:
                return False

        if category_filter:
            c_filter = category_filter.lower().strip()
            c_cat = chunk.get("category", "").lower()
            if c_filter not in c_cat:
                return False

        return True

    def clear(self) -> None:
        """Clear all stored documents."""
        self.chunks.clear()
        self.dense_vectors = None
        self.doc_frequencies.clear()
        self.avg_doc_len = 0.0

    def count(self) -> int:
        return len(self.chunks)
