# idea_saturation_embeddings.py — OpenAI‑python ≥ 1.0 compatible ────────────────
"""
Detect the request at which a *creative plateau* is reached – i.e. when
no *novel* ideas are produced for a configurable number of consecutive
requests – using OpenAI’s **`text-embedding-3-small`** model *via the new
`openai‑python ≥ 1.0` SDK*.

Quick‑start
───────────
```bash
pip install --upgrade "openai>=1.0.0" tqdm numpy pandas scikit-learn sqlalchemy
export OPENAI_API_KEY="sk‑…"   # or set OPENAI_ORG / OPENAI_BASE_URL as needed
python idea_saturation_embeddings.py
```

Main changes vs. the legacy SDK version
──────────────────────────────────────
* Import is now `from openai import OpenAI`; you instantiate a **client**.
* Endpoint is `client.embeddings.create()` and returns objects instead of
  dicts.
* Everything else – batching, cosine‑similarity freshness test, plateau
  detection – is unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("automation.env")

# ------------------------------------------------------------------ #
# 0 ▸ OpenAI client (new SDK)                                         #
# ------------------------------------------------------------------ #
try:
    import openai  # ≥1.0.0
except ImportError as e:
    raise SystemExit(
        "`openai>=1.0.0` required. Install/upgrade with:\n"
        "pip install --upgrade openai\n"
    ) from e

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL: str = "text-embedding-3-small"  # 1536‑d vectors
BATCH_SIZE: int = 100                         # safe for ≈8000 tokens
SIM_THRESHOLD: float = 0.6                   # cosine similarity cutoff
WINDOW: int = 1                               # # stagnating requests → plateau

# ------------------------------------------------------------------ #
# 1 ▸ Load conversation data                                          #
# ------------------------------------------------------------------ #
from database import engine, Request, Response  # project‑local module

with Session(engine) as s:
    rows = s.execute(
        select(
            Request.id.label("request_id"),
            Request.model.label("model"),
            Response.bullet_point.label("idea"),
        ).join(Response)
        .order_by(Request.id, Response.bullet_number)
    ).all()

df = pd.DataFrame(rows)
idea_texts: List[str] = df["idea"].tolist()

# ------------------------------------------------------------------ #
# 2 ▸ Fetch embeddings in mini‑batches                                #
# ------------------------------------------------------------------ #
print(f"Requesting {len(idea_texts):,} embeddings from OpenAI …")
embeddings: List[np.ndarray] = []

for i in tqdm(range(0, len(idea_texts), BATCH_SIZE), desc="embedding"):
    batch = [t.replace("\n", " ") for t in idea_texts[i : i + BATCH_SIZE]]
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    embeddings.extend([np.asarray(obj.embedding, dtype=np.float32) for obj in resp.data])

assert len(embeddings) == len(idea_texts)

df["embedding"] = embeddings

# ------------------------------------------------------------------ #
# 3 ▸ Plateau‑detection routine                                       #
# ------------------------------------------------------------------ #

def plateau(sub: pd.DataFrame, window: int, sim_thr: float) -> Tuple[int | None, int]:
    """Return (request_id_of_plateau, n_distinct_ideas)."""
    seen_embs: List[np.ndarray] = []
    stagn: int = 0

    for rid, ideas in sub.groupby("request_id"):
        fresh_found = False
        for emb in ideas["embedding"]:
            if not seen_embs:
                fresh_found = True
            else:
                sims = cosine_similarity([emb], seen_embs)[0]
                if np.all(sims < sim_thr):
                    fresh_found = True
            if fresh_found:
                seen_embs.append(emb)
        stagn = 0 if fresh_found else stagn + 1
        if stagn >= window:
            return rid, len(seen_embs)
    return None, len(seen_embs)

# ------------------------------------------------------------------ #
# 4 ▸ Process each generation model                                   #
# ------------------------------------------------------------------ #
result: Dict[str, Dict[str, int | None]] = defaultdict(dict)

for model, sub in tqdm(df.groupby("model"), desc="models"):
    req, total = plateau(sub, window=WINDOW, sim_thr=SIM_THRESHOLD)
    result[model]["plateau_request"] = req
    result[model]["distinct_ideas"] = total

summary = (
    pd.DataFrame(result).T
      .sort_values("plateau_request", na_position="last")
)

print("\nSaturation summary (OpenAI embeddings):")
print(summary.to_markdown())

if __name__ == "__main__":
    print("\nDone. Adjust SIM_THRESHOLD / WINDOW as needed and rerun.")
