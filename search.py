"""
search.py — Exa neural search module.

Exposes four functions:
    fetch_articles()     — main topic search
    fetch_similar()      — find articles similar to a given URL (Exa-unique)
    get_topic_answer()   — AI-generated answer summary for a topic
    _parse_results()     — shared result normalisation

All Exa-specific types are contained here; nothing else in the project
imports exa-py directly.
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

# Maps UI labels → Exa search type strings
SEARCH_TYPES = {
    "Auto (recommended)": "auto",
    "Fast": "fast",
    "Deep": "deep",
}

# Maps UI labels → Exa category strings (None = no category filter)
CATEGORIES = {
    "General": None,
    "News": "news",
    "Research Papers": "research paper",
}

# Maps UI labels → days to look back (None = no date filter)
RECENCY = {
    "All time": None,
    "Last month": 30,
    "Last week": 7,
    "Last 24 hours": 1,
}


# ── Client ─────────────────────────────────────────────────────────────────────

def _get_client():
    """Return an initialised Exa client, raising clearly if the key is missing."""
    from exa_py import Exa

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "EXA_API_KEY is not set. Add it to your .env file or environment."
        )
    return Exa(api_key=api_key)


# ── Result normalisation ───────────────────────────────────────────────────────

def _parse_results(results) -> list[dict]:
    """
    Normalise a list of Exa Result objects into plain dicts.

    Keys: id, title, url, published_date, text, highlights
    Falls back to title when body text is missing so embedding always works.
    """
    articles = []
    for r in results:
        text = (r.text or "").strip()
        title = (r.title or "").strip()

        # Highlights: list of excerpt strings, or empty list
        raw_highlights = getattr(r, "highlights", None) or []
        highlights = [h.strip() for h in raw_highlights if h and h.strip()]

        articles.append(
            {
                "id": r.id or r.url,
                "title": title or r.url,
                "url": r.url,
                "published_date": r.published_date or "Unknown",
                "text": text if text else title,
                "highlights": highlights,
            }
        )
    return articles


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_articles(
    query: str,
    num_results: int = 10,
    search_type: str = "auto",
    category: str | None = None,
    days_back: int | None = None,
) -> list[dict]:
    """
    Search Exa for *query* and return normalised article dicts.

    Args:
        query:        The search query.
        num_results:  How many results to request from Exa.
        search_type:  "fast" | "auto" | "deep"
        category:     "news" | "research paper" | None (= general web)
        days_back:    Only return articles published within this many days.
                      None = no date restriction.

    Raises:
        EnvironmentError  — missing API key
        RuntimeError      — Exa API call failed
    """
    client = _get_client()

    kwargs: dict = {
        "type": search_type,
        "num_results": num_results,
        "text": {"max_characters": 5000},
        "highlights": {"num_sentences": 3, "highlights_per_url": 1},
    }

    if category:
        kwargs["category"] = category

    if days_back:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
        kwargs["start_published_date"] = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        response = client.search_and_contents(query, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Exa search failed: {exc}") from exc

    return _parse_results(response.results)


def fetch_similar(url: str, num_results: int = 5) -> list[dict]:
    """
    Return articles semantically similar to *url* using Exa's find_similar
    endpoint — Exa's most distinctive capability.

    Args:
        url:         The source article URL to find neighbours for.
        num_results: How many similar articles to return.

    Raises:
        RuntimeError — Exa API call failed
    """
    client = _get_client()

    try:
        response = client.find_similar_and_contents(
            url,
            num_results=num_results,
            text={"max_characters": 5000},
            highlights={"num_sentences": 3, "highlights_per_url": 1},
        )
    except Exception as exc:
        raise RuntimeError(f"Exa find_similar failed: {exc}") from exc

    return _parse_results(response.results)


def get_topic_answer(query: str) -> str | None:
    """
    Return a short AI-generated answer about *query* with web citations,
    using Exa's /answer endpoint.

    Returns None gracefully if the endpoint is unavailable or fails,
    so the rest of the app still works.
    """
    client = _get_client()

    try:
        response = client.answer(query, text=True)
        return getattr(response, "answer", None) or str(response)
    except Exception:
        return None
