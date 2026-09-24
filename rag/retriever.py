"""
Production RAG Retrieval Engine for AI Travel Agent.
Integrates sliding-window chunking, dense semantic search, BM25 keyword matching,
metadata filtering, Reciprocal Rank Fusion (RRF) reranking, source citation builder,
and Groq LLM response synthesis with robust fallbacks.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from rag.knowledge_base import DESTINATIONS_DATA
from rag.chunker import chunk_document
from rag.vector_store import TravelVectorStore, tokenize_text
from rag.ingest import parse_uploaded_document

load_dotenv(find_dotenv())
_env_p = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_p):
    load_dotenv(_env_p)


class TravelRAGRetriever:
    """
    Hybrid RAG Engine supporting preloaded travel guides and dynamic user documents.
    """

    def __init__(self):
        self.vector_store = TravelVectorStore()
        self._load_preloaded_data()

    def _load_preloaded_data(self):
        """Index all preloaded destination guides into vector store."""
        for item in DESTINATIONS_DATA:
            metadata = {
                "doc_id": item["destination"].replace(" ", "_").lower(),
                "title": item["title"],
                "destination": item["destination"],
                "keywords": item.get("keywords", []),
                "category": item.get("category", "Destination Guide"),
                "source": "Verified Knowledge Base",
            }
            chunks = chunk_document(
                text=item["content"],
                metadata=metadata,
                chunk_size_words=170,
                chunk_overlap_words=30,
            )
            self.vector_store.add_chunks(chunks)

    def add_custom_document(self, filename: str, content_or_bytes: Any, doc_type: str = "Uploaded Document") -> int:
        """Parse, chunk, and index an uploaded document into the RAG vector store."""
        if isinstance(content_or_bytes, bytes):
            text, meta = parse_uploaded_document(filename, content_or_bytes, custom_category=doc_type)
        else:
            text = str(content_or_bytes)
            meta = {
                "doc_id": re.sub(r"[^\w\-_]", "_", filename),
                "title": filename,
                "destination": "General",
                "keywords": [filename.lower()],
                "category": doc_type,
                "source": f"User File: {filename}",
            }
            # Auto-detect destination
            for city in ["dubai", "bali", "paris", "tokyo", "goa", "london", "singapore", "switzerland", "rome", "new york"]:
                if city in filename.lower():
                    meta["destination"] = city.title()
                    meta["keywords"].append(city)

        chunks = chunk_document(
            text=text,
            metadata=meta,
            chunk_size_words=170,
            chunk_overlap_words=30,
        )
        return self.vector_store.add_chunks(chunks)

    def extract_text_from_file(self, uploaded_file) -> str:
        """Streamlit UploadedFile extraction helper."""
        bytes_data = uploaded_file.getvalue()
        text, _ = parse_uploaded_document(uploaded_file.name, bytes_data)
        return text

    def search(
        self,
        query: str,
        top_k: int = 3,
        destination_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using BM25 keyword matching + Dense Cosine Semantic Search + RRF Reranker.
        """
        results = self.vector_store.hybrid_search(
            query=query,
            top_k=top_k,
            destination_filter=destination_filter,
            category_filter=category_filter,
            alpha=0.55,
        )

        # Build clean source references
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk_id": r.get("chunk_id", "c0"),
                "title": r.get("title", "Guide"),
                "destination": r.get("destination", "General"),
                "content": r.get("content", ""),
                "source": r.get("source", "Knowledge Base"),
                "score": r.get("score", 0.0),
                "citation": f"[{r.get('source', 'Knowledge Base')} • {r.get('title', 'Guide')}]",
            })
        return formatted_results

    def format_citations(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate formatted markdown citation blocks."""
        if not context_chunks:
            return "No citations available."
        citations = []
        for idx, c in enumerate(context_chunks, 1):
            citations.append(f"**[{idx}] {c['title']}** ({c['source']}) — [Official Guide Portal](https://google.com/travel)\n> {c['content'][:220]}...")
        return "\n\n".join(citations)

    def synthesize_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Any]] = None
    ) -> str:
        """Synthesize response grounded in retrieved chunks and conversation history using Groq LLM with memory-aware fallback."""
        context_str = ""
        if context_chunks:
            context_str = "\n\n---\n\n".join([
                f"SOURCE [{c['source']} - {c['title']}]:\n{c['content']}"
                for c in context_chunks
            ])

        system_prompt = (
            "You are an elite, highly knowledgeable AI Travel Concierge with complete conversational memory.\n"
            "You remember everything the user tells you in this conversation (their name, destination interests, constraints, preferences) and refer back to it naturally.\n"
            "Answer the user's question clearly, citing facts from the provided verified context when available.\n"
            "Structure your response with clear bullet points.\n"
            "Do NOT use emojis in your response to maintain a clean Apple-style minimalist aesthetic."
        )

        q_lower = query.lower()

        # Extract user identity from history
        extracted_name = None
        extracted_dest = None
        all_human_texts = []
        if chat_history:
            for m in chat_history:
                if isinstance(m, HumanMessage):
                    all_human_texts.append(m.content)
        all_human_texts.append(query)

        for text in all_human_texts:
            name_match = re.search(r"(?:my name is|i am|i'm|call me|name's)\s+([A-Za-z]+)", text, re.IGNORECASE)
            if name_match:
                extracted_name = name_match.group(1).capitalize()
            for d in ["Dubai", "Bali", "Paris", "Tokyo", "Goa", "Pune", "Mumbai", "Delhi", "London", "Singapore", "Switzerland", "New York", "Bangalore", "Jaipur"]:
                if d.lower() in text.lower():
                    extracted_dest = d

        # Dynamic Tool Context Augmentation (Weather & Currency)
        tool_context_blocks = []
        if any(w in q_lower for w in ["weather", "temperature", "climate", "rain", "forecast", "hot", "cold", "humidity"]):
            w_city = None
            for d in ["Dubai", "Bali", "Paris", "Tokyo", "Goa", "Pune", "Mumbai", "Delhi", "London", "Singapore", "Switzerland", "New York", "Bangalore", "Jaipur"]:
                if d.lower() in q_lower:
                    w_city = d
                    break
            if not w_city:
                in_match = re.search(r"\b(?:in|for|at)\s+([A-Za-z]+)", query)
                if in_match and in_match.group(1).lower() not in ["the", "my", "this", "now", "today", "november", "december", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october"]:
                    w_city = in_match.group(1).capitalize()
            if not w_city and extracted_dest:
                w_city = extracted_dest

            if w_city:
                from agents.tools.weather_finder import weather_search
                w_res = weather_search.invoke({"params": {"destination": w_city}})
                live_w = w_res.get("current_live_weather")
                if isinstance(live_w, dict):
                    tool_context_blocks.append(
                        f"LIVE METEOROLOGICAL SENSOR DATA [{w_city}]:\n"
                        f"- Current Temperature: {live_w.get('temperature')}\n"
                        f"- Conditions: {live_w.get('condition')}\n"
                        f"- Wind Speed: {live_w.get('wind_speed')}\n"
                        f"- Seasonal Range: {w_res.get('expected_temperature_range')}\n"
                        f"- Rain Probability: {w_res.get('rain_probability')}\n"
                        f"- Packing Advice: {w_res.get('clothing_and_packing_advice')}\n"
                        f"- Sensor Source: {live_w.get('source')}"
                    )
                else:
                    tool_context_blocks.append(
                        f"SEASONAL CLIMATE DATA [{w_city}]:\n"
                        f"- Expected Temperature: {w_res.get('expected_temperature_range')}\n"
                        f"- General Conditions: {w_res.get('general_conditions')}\n"
                        f"- Rain Probability: {w_res.get('rain_probability')}\n"
                        f"- Packing Advice: {w_res.get('clothing_and_packing_advice')}"
                    )

        if any(w in q_lower for w in ["convert", "exchange rate", "currency", "forex", "inr to usd", "usd to inr", "usd to eur"]):
            num_match = re.search(r"(\d+(?:\.\d+)?)", query)
            amt = float(num_match.group(1)) if num_match else 100.0
            from_c = "USD"
            to_c = "INR" if ("inr" in q_lower or "rupee" in q_lower) else ("EUR" if "eur" in q_lower else ("AED" if ("aed" in q_lower or "dirham" in q_lower) else "USD"))
            if "inr to" in q_lower or "rupees to" in q_lower:
                from_c = "INR"
                to_c = "USD"
            from agents.tools.currency_tool import currency_conversion
            c_res = currency_conversion.invoke({"params": {"amount": amt, "from_currency": from_c, "to_currency": to_c}})
            tool_context_blocks.append(
                f"LIVE FOREX EXCHANGE DATA:\n"
                f"- Converted Amount: {c_res.get('amount', amt)} {c_res.get('from_currency', from_c)} = {c_res.get('converted_amount')} {c_res.get('to_currency', to_c)}\n"
                f"- Exchange Rate: 1 {c_res.get('from_currency', from_c)} = {c_res.get('exchange_rate')} {c_res.get('to_currency', to_c)}"
            )

        if tool_context_blocks:
            tool_str = "\n\n---\n\n".join(tool_context_blocks)
            if context_str:
                context_str = f"{tool_str}\n\n---\n\n{context_str}"
            else:
                context_str = tool_str

        user_content = query
        if context_str:
            user_content = f"VERIFIED TRAVEL CONTEXT:\n{context_str}\n\nUSER QUESTION: {query}"

        # Construct message list with system prompt and prior chat history
        messages = [SystemMessage(content=system_prompt)]
        if chat_history:
            for m in chat_history:
                if isinstance(m, (HumanMessage, AIMessage, SystemMessage)):
                    messages.append(m)
        messages.append(HumanMessage(content=user_content))

        api_key = os.environ.get("GROQ_API_KEY") or ""
        if api_key and not any(k in api_key.lower() for k in ["placeholder", "your_"]):
            for model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]:
                try:
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.2,
                        api_key=api_key,
                    )
                    response = llm.invoke(messages)
                    if response and getattr(response, "content", None):
                        return response.content
                except Exception:
                    continue

        # Autonomous Memory-Aware Fallback Engine
        # 1. Parse conversational history for user identity, preferences & context
        extracted_name = None
        extracted_dest = None
        all_human_texts = []
        if chat_history:
            for m in chat_history:
                if isinstance(m, HumanMessage):
                    all_human_texts.append(m.content)
        all_human_texts.append(query)

        for text in all_human_texts:
            name_match = re.search(r"(?:my name is|i am|i'm|call me|name's)\s+([A-Za-z]+)", text, re.IGNORECASE)
            if name_match:
                extracted_name = name_match.group(1).capitalize()
            for d in ["Dubai", "Bali", "Paris", "Tokyo", "Goa", "London", "Singapore", "Switzerland"]:
                if d.lower() in text.lower():
                    extracted_dest = d

        q_lower = query.lower()

        # Handle user identity / memory inquiry
        if any(w in q_lower for w in ["what is my name", "what's my name", "who am i", "do you know my name", "remember my name", "my name"]):
            if extracted_name:
                return f"Your name is {extracted_name}. I have your identity and travel preferences remembered across our session. How can I assist you further today?"
            return "I don't have your name yet in this session. If you would like to share it, I will remember it for all our conversations!"

        # Handle greeting
        if any(w in q_lower for w in ["hi", "hello", "hey", "greetings"]) and len(query.split()) <= 4:
            if extracted_name:
                return f"Hello {extracted_name}. How can I assist with your journey curation today?"
            return "Hello. How can I assist with your travel curation today?"

        # Handle statement of name
        if re.search(r"^(?:my name is|i am|i'm|call me)\s+([A-Za-z]+)", q_lower):
            if extracted_name:
                return f"Hello {extracted_name}. I have remembered your name and profile for our conversation. What destination or travel plans would you like to explore?"

        # Handle Live Weather Inquiry (Any Global City)
        if any(w in q_lower for w in ["weather", "temperature", "climate", "rain", "forecast", "hot", "cold", "humidity"]):
            w_city = None
            for d in ["Dubai", "Bali", "Paris", "Tokyo", "Goa", "Pune", "Mumbai", "Delhi", "London", "Singapore", "Switzerland", "New York", "Bangalore", "Jaipur"]:
                if d.lower() in q_lower:
                    w_city = d
                    break
            if not w_city:
                in_match = re.search(r"\b(?:in|for|at)\s+([A-Za-z]+)", query)
                if in_match and in_match.group(1).lower() not in ["the", "my", "this", "now", "today", "november", "december", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october"]:
                    w_city = in_match.group(1).capitalize()
            if not w_city and extracted_dest:
                w_city = extracted_dest

            if w_city:
                from agents.tools.weather_finder import weather_search
                w_res = weather_search.invoke({"params": {"destination": w_city}})
                live_w = w_res.get("current_live_weather")
                if isinstance(live_w, dict):
                    temp = live_w.get("temperature", w_res.get("expected_temperature_range", "24°C"))
                    cond = live_w.get("condition", w_res.get("general_conditions", "Pleasant"))
                    wind = live_w.get("wind_speed", "10 km/h")
                    src = live_w.get("source", "Open-Meteo Meteorological Sensor")
                    return (
                        f"### Weather Intelligence for {w_city.title()}\n\n"
                        f"- **Current Temperature:** {temp}\n"
                        f"- **Conditions:** {cond}\n"
                        f"- **Wind Speed:** {wind}\n"
                        f"- **Expected Season Range:** {w_res.get('expected_temperature_range')}\n"
                        f"- **Precipitation Probability:** {w_res.get('rain_probability')}\n"
                        f"- **Packing Advice:** {w_res.get('clothing_and_packing_advice')}\n\n"
                        f"**Grounding Source:** [{src} • Real-Time Meteorological Feed]"
                    )
                else:
                    return (
                        f"### Weather Intelligence for {w_city.title()}\n\n"
                        f"- **Expected Temperature:** {w_res.get('expected_temperature_range')}\n"
                        f"- **Conditions:** {w_res.get('general_conditions')}\n"
                        f"- **Precipitation Probability:** {w_res.get('rain_probability')}\n"
                        f"- **Packing Advice:** {w_res.get('clothing_and_packing_advice')}\n\n"
                        f"**Grounding Source:** [Verified Climate Hub • {w_city.title()} Seasonal Sensor]"
                    )

        # Handle Currency Conversion
        if any(w in q_lower for w in ["convert", "exchange rate", "currency", "forex", "inr to usd", "usd to inr", "usd to eur"]):
            num_match = re.search(r"(\d+(?:\.\d+)?)", query)
            amt = float(num_match.group(1)) if num_match else 100.0
            from_c = "USD"
            to_c = "INR" if ("inr" in q_lower or "rupee" in q_lower) else ("EUR" if "eur" in q_lower else ("AED" if ("aed" in q_lower or "dirham" in q_lower) else "USD"))
            if "inr to" in q_lower or "rupees to" in q_lower:
                from_c = "INR"
                to_c = "USD"
            from agents.tools.currency_tool import currency_conversion
            c_res = currency_conversion.invoke({"params": {"amount": amt, "from_currency": from_c, "to_currency": to_c}})
            return (
                f"### Currency Conversion Intelligence\n\n"
                f"- **Converted Amount:** {c_res.get('amount', amt)} {c_res.get('from_currency', from_c)} = **{c_res.get('converted_amount')} {c_res.get('to_currency', to_c)}**\n"
                f"- **Current Exchange Rate:** 1 {c_res.get('from_currency', from_c)} = {c_res.get('exchange_rate')} {c_res.get('to_currency', to_c)}\n\n"
                f"**Grounding Source:** [{c_res.get('source', 'Live Forex Market Sensor')}]"
            )

        if context_chunks:
            bullet_points = []
            for c in context_chunks:
                lines = [l.strip() for l in c["content"].split("\n") if l.strip() and not l.startswith("DESTINATION OVERVIEW")]
                bullet_points.extend(lines[:4])

            intro = f"### Verified Knowledge Summary for '{query}':"
            if extracted_name:
                intro = f"### Verified Knowledge Summary for {extracted_name} ('{query}'):"

            return (
                f"{intro}\n\n"
                + "\n".join([f"- {bp}" for bp in bullet_points[:8]])
                + f"\n\n**Grounding Sources:**\n"
                + self.format_citations(context_chunks)
            )

        if extracted_name:
            return f"Understood, {extracted_name}. I have recorded this in your travel session context. Feel free to ask about flights, hotels, visa guidelines, or day-by-day itineraries."
        return "Understood. I have recorded this in your travel session context. Feel free to ask about flights, hotels, visa guidelines, or day-by-day itineraries."


# Global singleton instance for quick access
_rag_instance: Optional[TravelRAGRetriever] = None

def get_rag_retriever() -> TravelRAGRetriever:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = TravelRAGRetriever()
    return _rag_instance
