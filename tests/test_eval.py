"""
Unit tests for Automated RAG & Agent Evaluation Suite.
Tests retrieval relevance, faithfulness, citations, tool selection,
hallucination risk, and composite benchmark metrics.
"""

import pytest
from eval.evaluator import RAGEvaluator


def test_retrieval_relevance_evaluation():
    query = "Dubai visa rules for Indian passport holder"
    chunks = [
        {"title": "Dubai Visa Guide", "content": "Indian passport holders with US visa get visa on arrival in Dubai."},
        {"title": "Dubai Sightseeing", "content": "Burj Khalifa and Dubai Mall are iconic attractions."},
    ]
    eval_res = RAGEvaluator.evaluate_retrieval_relevance(query, chunks, expected_keywords=["visa", "arrival", "passport"])
    assert eval_res["precision_at_k"] >= 0.5
    assert eval_res["semantic_relevance_score"] >= 50.0
    assert eval_res["status"] in ["EXCELLENT", "ACCEPTABLE"]


def test_answer_faithfulness_evaluation():
    chunks = [
        {"title": "Bali Guide", "content": "The Bali tourist levy is 150,000 IDR per person payable via Love Bali app."}
    ]
    grounded_answer = "The tourist levy fee in Bali is 150,000 IDR and should be paid through the Love Bali app."
    faith_res = RAGEvaluator.evaluate_answer_faithfulness(grounded_answer, chunks)
    assert faith_res["faithfulness_score"] >= 70.0
    assert faith_res["status"] in ["HIGHLY_FAITHFUL", "MODERATE_GROUNDING"]


def test_citation_correctness_evaluation():
    chunks = [{"title": "Paris Schengen Guide", "destination": "Paris"}]
    answer_with_citations = "Schengen visa requires €30,000 insurance. [Verified Travel Guide • Paris Schengen Guide] [🔗 Book Flight](https://google.com)"
    cite_res = RAGEvaluator.evaluate_citation_correctness(answer_with_citations, chunks)
    assert cite_res["citation_score"] >= 70.0
    assert cite_res["has_formal_citations"] is True
    assert cite_res["has_booking_links"] is True


def test_tool_selection_accuracy():
    selected = ["flights_finder", "hotels_finder", "travel_knowledge_search"]
    expected = ["flights_finder", "hotels_finder", "travel_knowledge_search"]
    tool_eval = RAGEvaluator.evaluate_tool_selection(selected, expected)
    assert tool_eval["tool_f1_score"] == 1.0
    assert tool_eval["status"] == "OPTIMAL"


def test_hallucination_detection():
    chunks = [{"title": "Dubai Guide", "content": "Rove hotel costs around $145 per night."}]
    safe_text = "The estimated hotel rate is $145 per night. Flights are around $310."
    hal_res = RAGEvaluator.evaluate_hallucination_risk(safe_text, chunks, known_prices=[145.0, 310.0])
    assert hal_res["hallucination_safety_score"] >= 80.0
    assert hal_res["risk_level"] == "LOW_RISK"


def test_run_benchmark_suite():
    bench = RAGEvaluator.run_benchmark_suite()
    assert "overall_composite_score" in bench
    assert bench["overall_composite_score"] >= 60.0
    assert bench["test_cases_evaluated"] >= 4
    assert bench["verdict"] in ["PRODUCTION_READY", "NEEDS_CALIBRATION"]
