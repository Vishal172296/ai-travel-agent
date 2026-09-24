"""
Document Ingestion & Parser for RAG Knowledge Hub.
Supports extracting and normalizing text from PDFs (baggage policies, vouchers, visa forms),
Markdown guides, plain text files, and structured JSON files.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def extract_text_from_pdf_stream(file_bytes: bytes) -> str:
    """Extract text from PDF binary bytes using pypdf with robust fallbacks."""
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        extracted_pages: List[str] = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                extracted_pages.append(f"--- PAGE {idx+1} ---\n{page_text}")
        if extracted_pages:
            return "\n\n".join(extracted_pages)
    except Exception:
        pass

    # Regex heuristic fallback for text streams in basic PDFs
    try:
        text_matches = re.findall(rb"\(([A-Za-z0-9 ,.\-!?_:/@#%&*+=\n]+)\)T[jd]", file_bytes)
        if text_matches:
            return " ".join(m.decode("latin1", errors="ignore") for m in text_matches)
    except Exception:
        pass

    return file_bytes.decode("utf-8", errors="ignore")


def parse_uploaded_document(
    filename: str,
    file_bytes: bytes,
    custom_category: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Parse uploaded file and extract text and structured metadata.
    Returns (clean_text, metadata_dict).
    """
    fn_lower = filename.lower()
    meta: Dict[str, Any] = {
        "filename": filename,
        "doc_id": re.sub(r"[^\w\-_]", "_", filename),
        "source": f"Uploaded File: {filename}",
        "category": custom_category or "User Document",
        "destination": "General",
        "keywords": [filename.lower()],
    }

    # Infer destination from filename if present
    for city in ["dubai", "bali", "paris", "tokyo", "goa", "london", "singapore", "switzerland", "rome", "new york"]:
        if city in fn_lower:
            meta["destination"] = city.title()
            meta["keywords"].append(city)

    # Extract text based on file format
    if fn_lower.endswith(".pdf"):
        meta["category"] = custom_category or "Travel PDF Document"
        text = extract_text_from_pdf_stream(file_bytes)
    elif fn_lower.endswith(".json"):
        meta["category"] = custom_category or "Structured JSON Guide"
        try:
            raw_data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
            if isinstance(raw_data, list):
                text = "\n\n".join([json.dumps(item, indent=2) for item in raw_data])
            else:
                text = json.dumps(raw_data, indent=2)
        except Exception:
            text = file_bytes.decode("utf-8", errors="ignore")
    elif fn_lower.endswith(".md") or fn_lower.endswith(".markdown"):
        meta["category"] = custom_category or "Markdown Guide"
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        # Default text
        text = file_bytes.decode("utf-8", errors="ignore")

    return text.strip(), meta
