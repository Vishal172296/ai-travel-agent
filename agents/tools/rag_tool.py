"""
LangChain tool for querying the Travel Knowledge Base / RAG Engine.
Allows the travel agent to fetch verified facts on visas, seasons, transit,
local culture, packing, and safety for destinations using hybrid search and metadata filtering.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from rag.retriever import get_rag_retriever
from agents.tools.cache import GLOBAL_TOOL_CACHE


class TravelGuideInput(BaseModel):
    query: str = Field(
        description="The topic or specific question to search (e.g., 'visa requirements', 'best time to visit', 'metro transit pass', 'cultural etiquette', 'baggage policy', 'safety rules')"
    )
    destination: Optional[str] = Field(
        default=None,
        description="The specific city or country name, e.g. 'Dubai', 'Bali', 'Paris', 'Tokyo', 'Goa', 'London', 'Singapore', 'Switzerland'"
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter, e.g. 'Destination Guide', 'Baggage Policy', 'Visa', 'General Guide'"
    )


class TravelGuideInputSchema(BaseModel):
    params: TravelGuideInput


@tool(args_schema=TravelGuideInputSchema)
def travel_knowledge_search(params: TravelGuideInput) -> str:
    """
    Search the Verified Travel Knowledge Base (RAG) for official visa regulations,
    entry rules, seasonal climate advice, transit hacks, cultural etiquette,
    baggage limits, and safety information with source citations.
    """
    cached = GLOBAL_TOOL_CACHE.get("travel_knowledge_search", params.dict())
    if cached:
        return cached

    retriever = get_rag_retriever()
    results = retriever.search(
        query=params.query,
        top_k=3,
        destination_filter=params.destination,
        category_filter=params.category,
    )

    if not results:
        res = "No specific guide found in the travel knowledge base for this query. Use general recommendations."
        GLOBAL_TOOL_CACHE.set("travel_knowledge_search", params.dict(), res)
        return res

    formatted = []
    for r in results:
        formatted.append(
            f"--- SOURCE: {r['source']} ({r['title']}) [Relevance Score: {r['score']}] ---\n{r['content']}"
        )

    res = "\n\n".join(formatted)
    GLOBAL_TOOL_CACHE.set("travel_knowledge_search", params.dict(), res)
    return res


# Backward compatible alias for existing pipelines
@tool(args_schema=TravelGuideInputSchema)
def search_travel_guide(params: TravelGuideInput) -> str:
    """
    Search the Travel Knowledge Base (RAG) for verified destination intelligence.
    (Alias for travel_knowledge_search).
    """
    return travel_knowledge_search.invoke({"params": params.dict()})
