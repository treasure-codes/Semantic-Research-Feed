"""
novelty.py — FAISS-based novelty detection.

Pure vector logic — no UI, no API calls.

Novelty score definition:
    score = 1.0 - max_cosine_similarity_to_any_seen_article

    1.0  → completely new (nothing similar has been seen)
    0.0  → identical to something already seen
"""

import numpy as np
import faiss

from embedder import EMBEDDING_DIM


def make_index() -> faiss.IndexFlatIP:
    """
    Create and return an empty FAISS IndexFlatIP.

    IndexFlatIP performs exhaustive inner-product search. When vectors are
    L2-normalised (as embedder.py guarantees), inner product == cosine similarity.
    """
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def score_novelty(index: faiss.IndexFlatIP, vector: np.ndarray) -> float:
    """
    Return the novelty score for a single vector against the current index.

    Args:
        index:  The FAISS index of already-seen article vectors.
        vector: 1-D or 2-D float32 array of shape (384,) or (1, 384).

    Returns:
        Float in [0.0, 1.0].
        Returns 1.0 (fully novel) when the index is empty.
    """
    if index.ntotal == 0:
        return 1.0

    vec = np.atleast_2d(vector).astype(np.float32)
    # k=1: we only need the single most-similar article
    distances, _ = index.search(vec, k=1)
    max_similarity = float(distances[0, 0])
    # Clamp to [0, 1] in case of floating-point overshoot
    max_similarity = max(0.0, min(1.0, max_similarity))
    return 1.0 - max_similarity


def add_vectors(index: faiss.IndexFlatIP, vectors: np.ndarray) -> None:
    """
    Add a batch of L2-normalised vectors to the index.

    Args:
        index:   The FAISS index to update (mutated in place).
        vectors: float32 array of shape (n, 384).
    """
    if vectors.shape[0] == 0:
        return
    index.add(vectors.astype(np.float32))
