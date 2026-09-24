"""
TypedDict State definition for the Agentic AI Travel Planner LangGraph Workflow.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    # Chat / message history with reducer
    messages: Annotated[List[AnyMessage], operator.add]

    # Agentic plan & execution tracking
    plan: Optional[List[str]]
    current_step: Optional[int]
    total_steps: Optional[int]

    # User Preferences
    preferences: Optional[Dict[str, Any]]

    # Trip specifics
    trip_details: Optional[Dict[str, Any]]

    # Observability & Debug Trace (safe for display without leaking private CoT)
    reasoning_trace: Annotated[List[Dict[str, Any]], operator.add]
    selected_tools: Annotated[List[str], operator.add]
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]

    # Human-in-the-loop (HITL) state
    approval_required: Optional[bool]
    approval_type: Optional[str]  # "flight_selection", "hotel_selection", "itinerary_approval", "budget_exceeded", "send_email"
    approval_status: Optional[str]  # "PENDING", "APPROVED", "MODIFIED", "REJECTED"
    human_feedback: Optional[str]  # e.g. "too expensive", "change Day 2"

    # Reflection & Validation state
    validation_status: Optional[str]  # "PASSED", "WARNING", "FAILED", "RETRIED"
    validation_details: Optional[Dict[str, Any]]
    retry_count: Optional[int]
