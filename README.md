# Semantic Research Feed

A real-time research discovery engine that uses neural search and semantic similarity to surface articles you haven't seen yet — powered by Exa's search API.

## What it does

You define a topic. The app continuously pulls fresh articles via Exa, embeds them, and compares them against your "already seen" vector index. Only semantically novel content breaks through — no duplicates, no retreads.

## How it works

1. **Search** — Exa API fetches live articles on your topic
2. **Embed** — sentence-transformers encodes each article into a vector
3. **Novelty check** — cosine similarity against FAISS index filters already-seen content
4. **Surface** — new, semantically distinct articles appear in the dashboard

## Stack

- `exa-py` — neural search API
- `sentence-transformers` — article embeddings
- `FAISS` — vector similarity search
- `Streamlit` — live dashboard
- `Python 3.11+`

## Setup
```bash
git clone https://github.com/yourusername/semantic-research-feed
cd semantic-research-feed
pip install -r requirements.txt
```

Add your API key:
```bash
export EXA_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Demo

[add gif or screenshot here once built]
