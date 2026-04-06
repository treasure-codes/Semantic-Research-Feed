"""
app.py — Semantic Research Feed
Streamlit frontend wiring together Exa search, sentence-transformer
embeddings, and FAISS novelty detection.
"""

import os
from datetime import datetime

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from embedder import build_embed_texts, embed_texts, load_model
from novelty import add_vectors, make_index, score_novelty
from search import (
    CATEGORIES,
    RECENCY,
    SEARCH_TYPES,
    fetch_articles,
    fetch_similar,
    get_topic_answer,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Research Feed",
    page_icon="🔭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .novelty-badge {
        border-radius: 8px;
        padding: 6px 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1rem;
        color: white;
    }
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────

if "index" not in st.session_state:
    st.session_state.index = make_index()
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids: set[str] = set()
if "novel_articles" not in st.session_state:
    st.session_state.novel_articles: list[dict] = []
if "last_stats" not in st.session_state:
    st.session_state.last_stats: tuple[int, int] | None = None
if "topic_answer" not in st.session_state:
    st.session_state.topic_answer: str | None = None
if "export_md" not in st.session_state:
    st.session_state.export_md: str = ""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _process_articles(articles: list[dict], threshold: float) -> tuple[list[dict], int]:
    """
    Embed, novelty-score, and filter a list of articles.
    Adds all vectors to the index (novel or not) so future searches filter correctly.
    Returns (novel_articles, total_new_to_index).
    """
    new_articles = [a for a in articles if a["id"] not in st.session_state.seen_ids]
    if not new_articles:
        return [], 0

    embed_strings = build_embed_texts(new_articles)
    vectors = embed_texts(embed_strings)

    novel = []
    for article, vector in zip(new_articles, vectors):
        article["novelty_score"] = score_novelty(st.session_state.index, vector)
        st.session_state.seen_ids.add(article["id"])
        if article["novelty_score"] >= threshold:
            novel.append(article)

    add_vectors(st.session_state.index, vectors)
    return novel, len(new_articles)


def _build_export_md(articles: list[dict], topic: str) -> str:
    lines = [f"# Semantic Research Feed — {topic}", f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]
    for a in articles:
        pct = int(a["novelty_score"] * 100)
        lines.append(f"## [{a['title']}]({a['url']})")
        lines.append(f"**Published:** {a['published_date']}  |  **Novelty:** {pct}% new\n")
        if a.get("highlights"):
            lines.append(f"> {a['highlights'][0]}\n")
        elif a.get("text") and a["text"] != a["title"]:
            lines.append(f"> {a['text'][:200]}…\n")
    return "\n".join(lines)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔭 Research Feed")
    st.caption("Surface only what's semantically new.")

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. multimodal LLMs, climate policy, CRISPR",
    )

    st.subheader("Search settings")

    search_type_label = st.selectbox("Search depth", list(SEARCH_TYPES.keys()), index=0)
    category_label = st.selectbox("Category", list(CATEGORIES.keys()), index=0)
    recency_label = st.selectbox("Recency", list(RECENCY.keys()), index=0)
    num_results = st.slider("Articles to fetch", min_value=5, max_value=30, value=10, step=5)

    st.subheader("Novelty filter")
    novelty_threshold = st.slider(
        "Minimum novelty score",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Higher = only very different articles surface. Lower = more permissive.",
    )

    search_clicked = st.button("Search", type="primary", use_container_width=True)

    st.divider()
    reset_clicked = st.button("Reset feed", use_container_width=True,
        help="Clears the seen-articles index so everything looks fresh again.")

    st.divider()
    st.caption(f"**Articles indexed:** {st.session_state.index.ntotal}")

# ── Reset ─────────────────────────────────────────────────────────────────────

if reset_clicked:
    st.session_state.index = make_index()
    st.session_state.seen_ids = set()
    st.session_state.novel_articles = []
    st.session_state.last_stats = None
    st.session_state.topic_answer = None
    st.session_state.export_md = ""
    st.success("Feed reset.")

# ── API key guard ─────────────────────────────────────────────────────────────

if not os.getenv("EXA_API_KEY"):
    st.error(
        "**EXA_API_KEY is not set.**\n\n"
        "Create a `.env` file in the project root:\n```\nEXA_API_KEY=your_key_here\n```"
    )
    st.stop()

# ── Warm up embedding model ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_model():
    return load_model()

get_model()

# ── Search handler ────────────────────────────────────────────────────────────

if search_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        search_type = SEARCH_TYPES[search_type_label]
        category = CATEGORIES[category_label]
        days_back = RECENCY[recency_label]

        # Fetch topic answer in parallel context (best-effort)
        with st.spinner("Getting topic summary from Exa…"):
            st.session_state.topic_answer = get_topic_answer(topic.strip())

        with st.spinner(f'Searching Exa for "{topic}" ({search_type_label})…'):
            try:
                articles = fetch_articles(
                    topic.strip(),
                    num_results=num_results,
                    search_type=search_type,
                    category=category,
                    days_back=days_back,
                )
            except (EnvironmentError, RuntimeError) as exc:
                st.error(str(exc))
                st.stop()

        if not articles:
            st.info("Exa returned no results. Try a different or broader topic.")
        else:
            with st.spinner("Embedding and scoring novelty…"):
                novel_batch, total_new = _process_articles(articles, novelty_threshold)

            if total_new == 0:
                st.info("All fetched articles were already seen. Try a different topic or reset.")
            else:
                st.session_state.novel_articles = novel_batch + st.session_state.novel_articles
                st.session_state.last_stats = (total_new, len(novel_batch))
                if st.session_state.novel_articles:
                    st.session_state.export_md = _build_export_md(
                        st.session_state.novel_articles, topic.strip()
                    )

# ── Main display ──────────────────────────────────────────────────────────────

st.title("Semantic Research Feed")

# Stats row
if st.session_state.last_stats:
    fetched, novel = st.session_state.last_stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetched", fetched)
    c2.metric("Novel", novel)
    c3.metric("Filtered out", fetched - novel)
    c4.metric("Total indexed", st.session_state.index.ntotal)
    st.divider()

# Topic answer
if st.session_state.topic_answer:
    st.markdown(
        f'<div class="answer-box"><strong>Topic summary</strong><br><br>{st.session_state.topic_answer}</div>',
        unsafe_allow_html=True,
    )

# Export button
if st.session_state.export_md:
    st.download_button(
        label="Export feed as Markdown",
        data=st.session_state.export_md,
        file_name="research_feed.md",
        mime="text/markdown",
    )

# No results yet
if not st.session_state.novel_articles:
    st.markdown(
        "Enter a topic in the sidebar and click **Search** to start discovering novel research."
    )
    st.stop()

# Article cards
for idx, article in enumerate(st.session_state.novel_articles):
    novelty_pct = int(article["novelty_score"] * 100)
    color = "#2ecc71" if novelty_pct >= 60 else "#f39c12" if novelty_pct >= 30 else "#e74c3c"

    with st.container(border=True):
        title_col, badge_col = st.columns([5, 1])

        with title_col:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.caption(f"Published: {article['published_date']}  ·  {article['url']}")

        with badge_col:
            st.markdown(
                f'<div class="novelty-badge" style="background:{color}">{novelty_pct}% new</div>',
                unsafe_allow_html=True,
            )

        # Prefer highlights (semantically extracted by Exa) over raw text
        if article.get("highlights"):
            for highlight in article["highlights"][:2]:
                st.markdown(f"> {highlight}")
        elif article.get("text") and article["text"] != article["title"]:
            st.markdown(f"> {article['text'][:300]}{'…' if len(article['text']) > 300 else ''}")

        # "Find Similar" — Exa's unique find_similar endpoint
        if st.button("Find similar articles", key=f"similar_{idx}_{article['id']}"):
            with st.spinner(f"Finding articles similar to this one…"):
                try:
                    similar = fetch_similar(article["url"], num_results=5)
                    novel_similar, _ = _process_articles(similar, novelty_threshold)
                except RuntimeError as exc:
                    st.error(str(exc))
                    novel_similar = []

            if novel_similar:
                st.session_state.novel_articles = (
                    novel_similar + st.session_state.novel_articles
                )
                st.session_state.export_md = _build_export_md(
                    st.session_state.novel_articles, "feed"
                )
                st.success(f"Added {len(novel_similar)} similar novel article(s). Scroll up to see them.")
                st.rerun()
            else:
                st.info("No novel similar articles found — they may already be in your feed.")
