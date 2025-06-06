#!/usr/bin/env python3
"""
embed_and_cluster_multi.py
--------------------------

* Fetches rows   →  embeds with text-embedding-3-small
* Clusters       →  K-Means **or** DBSCAN (or Hybrid)
* Persists back  →  writes embedding + chosen cluster id

Dependencies
------------
pip install openai pymysql scikit-learn numpy tqdm
"""

from __future__ import annotations
import argparse, json, math, os, time
from typing import List, Sequence

import numpy as np
import pymysql
from tqdm import tqdm
from openai import OpenAI
from sklearn.cluster import KMeans, DBSCAN
from dotenv import load_dotenv

# ─────────────────────────── DB CONFIG ────────────────────────────

load_dotenv("automation.env")

MYSQL_DSN = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "aut_experiment"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

TABLE          = "ideas_raw"
ID_FIELD       = "id"
TEXT_FIELD     = "bullet_point"
EMBED_FIELD    = "embedding"
CLUSTER_FIELD  = "cluster_id"

# ─────────────────── OPENAI & MISC CONSTANTS ──────────────────────
OPENAI_CLIENT = OpenAI(max_retries=5, timeout=60)
EMBED_MODEL   = "text-embedding-3-small"
DEFAULT_BATCH = 64
SLEEP_BETWEEN = 0.0                   

# ────────────────────────── UTILITIES ─────────────────────────────
def batched(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def fetch_rows(conn, only_new: bool) -> List[dict]:
    q  = f"SELECT {ID_FIELD}, {TEXT_FIELD}, {EMBED_FIELD} FROM {TABLE}"
    q += f" WHERE {TEXT_FIELD} IS NOT NULL"
    if only_new:
        q += f" AND {EMBED_FIELD} IS NULL"
    with conn.cursor() as cur:
        cur.execute(q)
        return list(cur.fetchall())

def embed_texts(texts: List[str], batch: int) -> List[List[float]]:
    out: List[List[float]] = []
    for chunk in tqdm(batched(texts, batch),
                      total=math.ceil(len(texts)/batch),
                      desc="Embedding", unit="batch"):
        resp = OPENAI_CLIENT.embeddings.create(input=chunk, model=EMBED_MODEL)
        out.extend(d.embedding for d in resp.data)
        time.sleep(SLEEP_BETWEEN)
    return out

# ────────────────────────── CLUSTERING ────────────────────────────
def kmeans_cluster(vectors: np.ndarray, k: int | None) -> List[int]:
    k = k or max(2, int(math.sqrt(len(vectors)/2)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    return km.fit_predict(vectors).tolist()

def dbscan_cluster(vectors: np.ndarray, eps: float, min_samples: int) -> List[int]:
    model = DBSCAN(eps=eps, metric="cosine",
                   min_samples=min_samples, n_jobs=-1)
    return model.fit_predict(vectors).tolist()

def hybrid_cluster(vectors: np.ndarray, coarse_k: int, eps: float) -> List[int]:
    """K-Means bucket → DBSCAN inside each bucket"""
    coarse = kmeans_cluster(vectors, coarse_k)
    fine   = np.full(len(vectors), -1, dtype=int)
    next_id = 0
    for c in np.unique(coarse):
        idx = np.where(np.array(coarse) == c)[0]
        if len(idx) < 2:  # nothing to cluster
            continue
        sub_labels = dbscan_cluster(vectors[idx], eps, 2)
        offset = max(sub_labels) + 1 if max(sub_labels) >= 0 else 0
        for i, lab in zip(idx, sub_labels):
            fine[i] = lab + next_id if lab >= 0 else -1
        next_id += offset
    return fine.tolist()

# ──────────────────────── WRITE-BACK ──────────────────────────────
def write_back(conn, ids: Sequence[int],
               embeds: Sequence[List[float]], labels: Sequence[int]):
    sql = (f"UPDATE {TABLE} SET {EMBED_FIELD} = %s, {CLUSTER_FIELD} = %s "
           f"WHERE {ID_FIELD} = %s")
    data = [(json.dumps(v), int(l), int(i))
            for i, v, l in zip(ids, embeds, labels)]
    with conn.cursor() as cur:
        cur.executemany(sql, data)
    conn.commit()

# ──────────────────────────── MAIN ────────────────────────────────
def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser("Embed & cluster (K-Means / DBSCAN / Hybrid)")
    ap.add_argument("--algo", choices=["kmeans", "dbscan", "hybrid"],
                    default="kmeans", help="Clustering backend")
    ap.add_argument("--k", type=int, default=None,
                    help="K for K-Means (coarse_K for hybrid)")
    ap.add_argument("--eps", type=float, default=0.20,
                    help="Cosine radius for DBSCAN / hybrid")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="DBSCAN min_samples")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="Embedding batch size")
    ap.add_argument("--only-new", action="store_true",
                    help="Skip rows that already have an embedding")
    args = ap.parse_args(argv)

    conn = pymysql.connect(**MYSQL_DSN)
    try:
        rows = fetch_rows(conn, args.only_new)
        if not rows:
            print("Nothing to process.")
            return

        ids   = [r[ID_FIELD]   for r in rows]
        texts = [r[TEXT_FIELD] for r in rows]

        embeds = embed_texts(texts, args.batch)
        X = np.asarray(embeds, dtype=np.float32)

        print(f"Clustering with {args.algo.upper()} …")
        if args.algo == "kmeans":
            labels = kmeans_cluster(X, args.k)
        elif args.algo == "dbscan":
            labels = dbscan_cluster(X, args.eps, args.min_samples)
        else:  # hybrid
            coarse_k = args.k or int(math.sqrt(len(X)))
            labels = hybrid_cluster(X, coarse_k, args.eps)

        write_back(conn, ids, embeds, labels)
        print("✔︎ Done.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
