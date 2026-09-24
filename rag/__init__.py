"""
RAG Package for AI Travel Planner.
"""

from rag.knowledge_base import DESTINATIONS_DATA
from rag.retriever import TravelRAGRetriever, get_rag_retriever
from rag.chunker import chunk_document
from rag.vector_store import TravelVectorStore
from rag.ingest import parse_uploaded_document
from rag.utils import get_map_dataframe, generate_ics_calendar, get_packing_checklist

__all__ = [
    "DESTINATIONS_DATA",
    "TravelRAGRetriever",
    "get_rag_retriever",
    "chunk_document",
    "TravelVectorStore",
    "parse_uploaded_document",
    "get_map_dataframe",
    "generate_ics_calendar",
    "get_packing_checklist",
]
