"""
Unit & Integration tests for the RAG Pipeline.
Tests chunking, vector store semantic search, BM25 keyword matching,
hybrid RRF retrieval, metadata filtering, custom document ingestion, and citations.
"""

import pytest
from rag.chunker import chunk_document, clean_text, split_into_sentences
from rag.vector_store import TravelVectorStore, tokenize_text
from rag.ingest import parse_uploaded_document
from rag.retriever import TravelRAGRetriever, get_rag_retriever


def test_clean_text_and_tokenization():
    raw = "  Hello   world!\r\n\r\nThis is Dubai, UAE.  "
    cleaned = clean_text(raw)
    assert "Hello world!" in cleaned
    tokens = tokenize_text(cleaned)
    assert "hello" in tokens
    assert "world" in tokens
    assert "dubai" in tokens
    assert "the" not in tokens  # Stop word removed


def test_chunking_with_overlap():
    sample_text = (
        "Dubai is a cosmopolitan hub in the UAE known for luxury shopping and desert safaris. "
        "The Burj Khalifa is the tallest building in the world. Visitors can enjoy the Dubai Fountain show. "
        "Visa on arrival is available for Indian passport holders with a valid US visa."
    )
    metadata = {"doc_id": "dubai_doc", "destination": "Dubai", "source": "Official Guide"}
    chunks = chunk_document(sample_text, metadata=metadata, chunk_size_words=15, chunk_overlap_words=5)

    assert len(chunks) >= 2
    assert all("chunk_id" in c for c in chunks)
    assert all(c["destination"] == "Dubai" for c in chunks)
    assert all(c["source"] == "Official Guide" for c in chunks)


def test_vector_store_hybrid_search():
    store = TravelVectorStore()
    doc1 = {
        "chunk_id": "c1",
        "destination": "Dubai",
        "category": "Destination Guide",
        "source": "Guide",
        "text": "Dubai visa requirements for Indian passport holders: Visa on arrival is available if you have a valid US visa.",
        "metadata": {"title": "Dubai Visa Guide", "keywords": ["dubai", "visa", "uae"]},
    }
    doc2 = {
        "chunk_id": "c2",
        "destination": "Bali",
        "category": "Destination Guide",
        "source": "Guide",
        "text": "Bali tourist levy fee is 150000 IDR payable via Love Bali app. Best time to visit is dry season from April to October.",
        "metadata": {"title": "Bali Travel Guide", "keywords": ["bali", "indonesia", "levy"]},
    }
    store.add_chunks([doc1, doc2])

    assert store.count() == 2

    # Query for Dubai
    dubai_results = store.hybrid_search("Dubai visa for Indian passport", top_k=1)
    assert len(dubai_results) == 1
    assert "Dubai" in dubai_results[0]["destination"]

    # Query for Bali
    bali_results = store.hybrid_search("Bali tourist levy Love Bali", top_k=1)
    assert len(bali_results) == 1
    assert "Bali" in bali_results[0]["destination"]

    # Test destination filtering
    filtered_results = store.hybrid_search("visa guide", top_k=2, destination_filter="Bali")
    for r in filtered_results:
        assert "Bali" in r["destination"]


def test_custom_document_ingestion():
    retriever = TravelRAGRetriever()
    initial_count = retriever.vector_store.count()

    custom_content = "Special airline baggage policy: Maximum 1 cabin bag of 7kg and 1 personal item. Power banks must not exceed 20000 mAh."
    added = retriever.add_custom_document("Airline_Baggage_Rules.txt", custom_content, doc_type="Baggage Policy")

    assert added >= 1
    assert retriever.vector_store.count() == initial_count + added

    search_res = retriever.search("power bank cabin baggage limits", top_k=2)
    assert len(search_res) >= 1
    assert any("Baggage" in r["title"] or "baggage" in r["content"].lower() for r in search_res)


def test_citations_formatting():
    retriever = get_rag_retriever()
    results = retriever.search("Paris Louvre Eiffel Tower", top_k=2)
    assert len(results) >= 1

    citations = retriever.format_citations(results)
    assert "[1]" in citations
    assert "Paris" in citations or "Guide" in citations


def test_chat_memory_retrieval():
    from langchain_core.messages import HumanMessage, AIMessage
    retriever = get_rag_retriever()

    history = [
        HumanMessage(content="my name is vishal"),
        AIMessage(content="Hello Vishal. How can I assist with your journey curation today?"),
    ]
    ans = retriever.synthesize_answer(
        query="what is my name?",
        context_chunks=[],
        chat_history=history,
    )
    assert "Vishal" in ans

