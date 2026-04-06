"""
embedder.py — Sentence-transformer embedding module.

The model is loaded once at module level. Streamlit reruns the script on
every interaction, but Python's module cache means the model object is only
instantiated once per process — no repeated downloads.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# 384-dimensional model; fast and accurate enough for this task.
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings and return L2-normalised float32 vectors.

    Shape: (len(texts), EMBEDDING_DIM)

    L2-normalisation is required so that inner-product search in FAISS
    equals cosine similarity.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = load_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype(np.float32)

    # Normalise in-place so cosine sim = inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors (shouldn't happen, but be safe)
    norms = np.where(norms == 0, 1.0, norms)
    vectors /= norms

    return vectors


def build_embed_texts(articles: list[dict]) -> list[str]:
    """
    Construct the string that gets embedded for each article.

    Combines title + first 500 chars of body text. Title carries the most
    topical signal; body text adds context without bloating token count.
    """
    texts = []
    for a in articles:
        title = a.get("title", "")
        body = a.get("text", "")[:500]
        texts.append(f"{title}. {body}".strip())
    return texts
