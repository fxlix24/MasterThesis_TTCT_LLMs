# idea_saturation_embeddings.py — OpenAI‑SDK ≥1.0, per‑model request index
"""
Compute where a *creative plateau* occurs for each model by tracking when
new ideas stop appearing (based on semantic similarity) **and** using a
per‑model request counter (`model_request_number`) instead of the global
`request_id`.

Updates in this revision
────────────────────────
* **OpenAI SDK ≥ 1.0** interface (`from openai import OpenAI`).
* SQLAlchemy query now annotates every request with a `ROW_NUMBER()` that
  resets for each model – mirroring your SQL snippet.
* Plateau logic:
    • Iterates over `model_request_number`.
    • Plateau is flagged after *N* consecutive *stagnant* requests, where
      *N = WINDOW* (default 1 ⇒ plateau only if the *current* request adds
      **no** new ideas).
* Summary prints `plateau_request_index` so it’s clear this is the
  per‑model counter.

Install deps & run
─────────────────
```bash
pip install --upgrade "openai>=1.0.0" tqdm numpy pandas scikit-learn sqlalchemy
export OPENAI_API_KEY="sk‑…"
python idea_saturation_embeddings.py
```
"""
from __future__ import annotations
import openai
import os

from collections import defaultdict
from typing import Dict, List, Tuple
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tqdm import tqdm

load_dotenv("automation.env")

# OpenAI client (≥1.0) ------------------------------------------------
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL: str = "text-embedding-3-small"  # 1536‑D
BATCH_SIZE: int = 100                         # safe default
SIM_THRESHOLD: float = 0.80                   # cosine similarity cutoff
WINDOW: int = 1                               # stagnant requests → plateau

# --------------------------------------------------------------------
# 1. Load conversation data WITH per‑model request index
# --------------------------------------------------------------------
from database import engine, Request, Response  # project‑local

model_req_no = func.row_number().over(
    partition_by=Request.model,
    order_by=Request.id,
).label("model_request_number")

with Session(engine) as s:
    rows = s.execute(
        select(
            Request.id.label("request_id"),
            model_req_no,
            Request.model.label("model"),
            Response.bullet_point.label("idea"),
        ).join(Response)
         .order_by(Request.model, model_req_no, Response.bullet_number)
    ).all()

df = pd.DataFrame(rows)
idea_texts: List[str] = df["idea"].tolist()

# --------------------------------------------------------------------
# 2. Fetch embeddings in mini‑batches
# --------------------------------------------------------------------
print(f"Requesting {len(idea_texts):,} embeddings from OpenAI …")
embeddings: List[np.ndarray] = []

for i in tqdm(range(0, len(idea_texts), BATCH_SIZE), desc="embedding"):
    batch = [t.replace("\n", " ") for t in idea_texts[i : i + BATCH_SIZE]]
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    embeddings.extend([np.asarray(obj.embedding, dtype=np.float32) for obj in resp.data])

assert len(embeddings) == len(idea_texts)

df["embedding"] = embeddings

# --------------------------------------------------------------------
# 3. Plateau‑detection routine (per‑model request index)
# --------------------------------------------------------------------

def plateau(sub: pd.DataFrame, window: int, sim_thr: float) -> Tuple[int | None, int]:
    """Return (**last** fresh `model_request_number`, n_distinct_ideas)."""
    seen_embs: List[np.ndarray] = []
    stagnant_streak: int = 0
    last_fresh_idx: int | None = None  # track most recent request that added ideas

    # Iterate in chronological order
    for idx, ideas in (
        sub.sort_values("model_request_number")
           .groupby("model_request_number")
    ):
        fresh_found = False
        for emb in ideas["embedding"].to_list():
            if not seen_embs:
                fresh_found = True
            else:
                if np.all(cosine_similarity([emb], seen_embs)[0] < sim_thr):
                    fresh_found = True
            if fresh_found:
                seen_embs.append(emb)
        if fresh_found:
            last_fresh_idx = idx
            stagnant_streak = 0
        else:
            stagnant_streak += 1
        if stagnant_streak >= window:  # plateau sustained
            return last_fresh_idx, len(seen_embs)
    # Plateau never reached
    return None, len(seen_embs)  # never plateaued

# --------------------------------------------------------------------
# 4. Process each generation model
# --------------------------------------------------------------------
result: Dict[str, Dict[str, int | None]] = defaultdict(dict)

for model, sub in tqdm(df.groupby("model"), desc="models"):
    idx, total = plateau(sub, window=WINDOW, sim_thr=SIM_THRESHOLD)
    result[model]["plateau_request_index"] = idx
    result[model]["distinct_ideas"] = total

summary = (
    pd.DataFrame(result).T
      .sort_values("plateau_request_index", na_position="last")
)

print("\nSaturation summary (plateau_request_index is per‑model):")
print(summary.to_markdown())

if __name__ == "__main__":
    print("\nDone. Adjust SIM_THRESHOLD / WINDOW as needed and rerun.")
