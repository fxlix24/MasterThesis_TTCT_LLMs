#!/usr/bin/env python3
"""
cluster_only.py  –  recluster *existing* embeddings

Usage examples
--------------
# DBSCAN with cosine radius 0.23
python cluster_only.py --algo dbscan  --eps 0.23

# K-Means, K = 300
python cluster_only.py --algo kmeans --k 300

# Hybrid:  √N buckets → DBSCAN (eps 0.22)
python cluster_only.py --algo hybrid --eps 0.22

# Hybrid but force 400 buckets
python cluster_only.py --algo hybrid --k 400 --eps 0.20
"""

import argparse, json, math, os
from typing import List, Tuple

import numpy as np
import pymysql
from sklearn.cluster import KMeans, DBSCAN
from dotenv import load_dotenv

# ────────────────────────  DB - SETTINGS  ────────────────────────

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
EMBED_FIELD    = "embedding"          # JSON column with stored vectors
CLUSTER_FIELD  = "cluster_id"         # BIGINT, signed

# ────────────────────────  HELPERS  ───────────────────────────────
def fetch_vectors(conn) -> Tuple[List[int], np.ndarray]:
    """Return (ids, 2-D numpy array of embeddings)."""
    sql = f"SELECT {ID_FIELD}, {EMBED_FIELD} FROM {TABLE} WHERE {EMBED_FIELD} IS NOT NULL"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    ids  = [r[ID_FIELD] for r in rows]
    vecs = [json.loads(r[EMBED_FIELD]) for r in rows]
    return ids, np.asarray(vecs, dtype=np.float32)

def write_labels(conn, ids: List[int], labels: List[int]):
    sql = f"UPDATE {TABLE} SET {CLUSTER_FIELD} = %s WHERE {ID_FIELD} = %s"
    with conn.cursor() as cur:
        cur.executemany(sql, list(zip(labels, ids)))
    conn.commit()

# ───────────────────  CLUSTERING BACK-ENDS  ───────────────────────
def kmeans_labels(X: np.ndarray, k: int | None) -> List[int]:
    k = k or max(2, int(math.sqrt(len(X) / 2)))
    return KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X).tolist()

def dbscan_labels(X: np.ndarray, eps: float, min_samples: int) -> List[int]:
    return DBSCAN(
        eps=eps,
        metric="cosine",
        min_samples=min_samples,
        n_jobs=-1,
    ).fit_predict(X).tolist()

def hybrid_labels(X: np.ndarray, coarse_k: int | None, eps: float) -> List[int]:
    """K-Means coarse buckets → DBSCAN inside each bucket."""
    coarse = kmeans_labels(X, coarse_k)
    fine   = np.full(len(X), -1, dtype=int)
    next_id = 0
    for bucket in np.unique(coarse):
        idx = np.where(np.array(coarse) == bucket)[0]
        if len(idx) < 2:
            continue
        local = dbscan_labels(X[idx], eps=eps, min_samples=2)
        offset = (max(local) + 1) if max(local) >= 0 else 0
        for i, lab in zip(idx, local):
            fine[i] = lab + next_id if lab >= 0 else -1
        next_id += offset
    return fine.tolist()

# ───────────────────────────  MAIN  ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Re-cluster embeddings already stored in MySQL")
    ap.add_argument("--algo", choices=["dbscan", "kmeans", "hybrid"],
                    default="dbscan", help="Clustering backend")
    ap.add_argument("--eps", type=float, default=0.22,
                    help="Cosine radius for DBSCAN / hybrid (0.15-0.30 typical)")
    ap.add_argument("--k", type=int, default=None,
                    help="K for K-Means (or coarse_k for hybrid); default √N/2")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="DBSCAN min_samples (ignored by K-Means / hybrid)")
    args = ap.parse_args()

    conn = pymysql.connect(**MYSQL_DSN)
    try:
        ids, X = fetch_vectors(conn)
        if not ids:
            print("No rows with stored embeddings—nothing to do.")
            return

        if args.algo == "kmeans":
            labels = kmeans_labels(X, args.k)
        elif args.algo == "dbscan":
            labels = dbscan_labels(X, eps=args.eps, min_samples=args.min_samples)
        else:                              # hybrid
            coarse_k = args.k or int(math.sqrt(len(X)))
            labels   = hybrid_labels(X, coarse_k, eps=args.eps)

        write_labels(conn, ids, labels)
        print(f"✓ Updated {len(ids):,} rows with new {args.algo} labels.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
