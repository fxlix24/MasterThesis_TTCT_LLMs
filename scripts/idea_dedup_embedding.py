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
import argparse, json, math, os, time, sys
from typing import List, Sequence

import numpy as np
import pymysql
from tqdm import tqdm
from openai import OpenAI
from idea_dedup_clustering import kmeans_labels, hybrid_labels, dbscan_labels
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from support_tools.ideas_sync import sync_ideas

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

EXPERIMENT_PHASE = 1 # Alternatively use 'os.getenv("PROJECT_PHASE")' --> returns Integer of currently selected PROJECT_PHASE
TABLE          = f"ideas_aut_{EXPERIMENT_PHASE}" 
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

def fetch_rows(conn, redo_all: bool) -> List[dict]:
    q  = f"SELECT {ID_FIELD}, {TEXT_FIELD}, {EMBED_FIELD} FROM {TABLE}"
    q += f" WHERE {TEXT_FIELD} IS NOT NULL"
    if not redo_all:
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
    ap.add_argument("--redo-all", action="store_true",
                    help="Redo embeddings for entire dataset (default: only process new rows)")
    args = ap.parse_args(argv)

    conn = pymysql.connect(**MYSQL_DSN)
    try:
        sync_ideas()
        rows = fetch_rows(conn, args.redo_all)
        if not rows:
            print("Nothing to process.")
            return

        ids   = [r[ID_FIELD]   for r in rows]
        texts = [r[TEXT_FIELD] for r in rows]

        embeds = embed_texts(texts, args.batch)
        X = np.asarray(embeds, dtype=np.float32)

        print(f"Clustering with {args.algo.upper()} …")
        if args.algo == "kmeans":
            labels = kmeans_labels(X, args.k)
        elif args.algo == "dbscan":
            labels = dbscan_labels(X, args.eps, args.min_samples)
        else:  # hybrid
            coarse_k = args.k or int(math.sqrt(len(X)))
            labels = hybrid_labels(X, coarse_k, args.eps)

        write_back(conn, ids, embeds, labels)
        print("✔︎ Done.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()