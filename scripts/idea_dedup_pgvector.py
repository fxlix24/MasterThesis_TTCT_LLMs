# idea_dedup_mysql.py ────────────────────────────────────────────────
"""
What it does – idempotently
───────────────────────────
1.  Connects to MySQL via `mysql-connector-python` (DSN in `MYSQL_DSN`).
2.  Adds missing columns / indexes (`embedding` JSON, `cluster_id`).
3.  Batch-embeds any rows where `embedding IS NULL` using OpenAI
    `text-embedding-3-small`.
4.  Loads the current embedding matrix into FAISS (HNSW) **once**.
5.  For each row with `cluster_id IS NULL`, finds its nearest neighbour:
    * If cosine similarity ≥ `SIM_THRESHOLD` (0.80) ⇒ duplicate ⇒ inherits
      neighbour’s `cluster_id`.
    * Else ⇒ novel ⇒ becomes its own cluster root (`cluster_id = id`).

Dependencies
────────────
```bash
pip install openai mysql-connector-python faiss-cpu tqdm numpy
```

Environment variables
─────────────────────
* `OPENAI_API_KEY` – your key.
* `MYSQL_DSN` (optional) – defaults to *localhost/aut_experiment*.

Run it whenever you ingest new ideas; already-processed rows are skipped.
"""
from __future__ import annotations

import json
import os
from itertools import islice
from typing import Iterable, List, Tuple

import faiss  # CPU build
import mysql.connector  as mysql
import numpy as np
import openai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("automation.env")

# ───────────────────────── CONFIGURATION ──────────────────────────────
MYSQL_DSN = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "aut_experiment"),
    "charset": "utf8mb4",
    "collation": "utf8mb4_unicode_ci",
}
EMBED_MODEL: str = "text-embedding-3-small"
BATCH_EMBED: int = 500          # keep < 8 191 tokens
SIM_THRESHOLD: float = 0.80

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────── HELPERS ────────────────────────────────────

def chunks(it: Iterable, n: int):
    "Yield successive n-sized chunks from *it*."
    it = iter(it)
    while chunk := list(islice(it, n)):
        yield chunk


# ───────────────────────── EMBED MISSING ROWS ─────────────────────────

def embed_missing(cur: mysql.cursor.MySQLCursor) -> None:
    cur.execute("SELECT id, bullet_text FROM ideas_raw WHERE embedding IS NULL;")
    rows = cur.fetchall()
    if not rows:
        print("No missing embeddings.")
        return

    print(f"Embedding {len(rows):,} ideas…")
    for chunk in tqdm(chunks(rows, BATCH_EMBED)):
        ids, texts = zip(*[(row[0], row[1].replace("\n", " ")) for row in chunk])
        resp = client.embeddings.create(model=EMBED_MODEL, input=list(texts))
        vecs = [json.dumps(e.embedding) for e in resp.data]
        cur.executemany(
            "UPDATE ideas_raw SET embedding = %s WHERE id = %s;",
            list(zip(vecs, ids)),
        )

# ───────────────────────── BUILD FAISS INDEX ─────────────────────────

def load_embeddings(cur: mysql.cursor.MySQLCursor, ids_only_null=False):
    query = "SELECT id, embedding FROM ideas_raw WHERE embedding IS NOT NULL"
    if ids_only_null:
        query += " AND cluster_id IS NULL"
    cur.execute(query)
    data = cur.fetchall()
    ids = []
    vecs = []
    for row in data:
        ids.append(row[0])
        vecs.append(np.asarray(json.loads(row[1]), dtype=np.float32))
    if not ids:
        return ids, None
    mat = np.vstack(vecs)
    d = mat.shape[1]
    index = faiss.IndexHNSWFlat(d, 64)
    index.hnsw.efSearch = 128
    index.add(mat)
    return ids, index

# ───────────────────────── DEDUP MISSING CLUSTERS ────────────────────

def dedup(cur: mysql.cursor.MySQLCursor) -> None:
    # Build full index once
    all_ids, index = load_embeddings(cur, ids_only_null=False)
    if index is None:
        print("No embeddings found – run embed step first.")
        return

    id_to_pos = {id_: pos for pos, id_ in enumerate(all_ids)}

    # Fetch rows that still lack cluster_id
    cur.execute("SELECT id, embedding FROM ideas_raw WHERE cluster_id IS NULL;")
    todo = cur.fetchall()
    if not todo:
        print("No rows left to cluster.")
        return

    updates: List[Tuple[int, int]] = []  # (cluster_id, id)

    for idea_id, emb_json in tqdm(todo, desc="dedup"):
        vec = np.asarray(json.loads(emb_json), dtype=np.float32)
        D, I = index.search(vec.reshape(1, -1), 2)  # self + nearest
        nn_pos = I[0, 1] if I.shape[1] > 1 else I[0, 0]
        nn_id = all_ids[nn_pos]
        cos_sim = 1 - D[0, 1]

        if cos_sim >= SIM_THRESHOLD and idea_id != nn_id:
            cluster_id = get_cluster(cur, nn_id)
        else:
            cluster_id = idea_id
        updates.append((cluster_id, idea_id))

    cur.executemany(
        "UPDATE ideas_raw SET cluster_id = %s WHERE id = %s;",
        updates,
    )


def get_cluster(cur: mysql.cursor.MySQLCursor, id_: int) -> int:
    cur.execute("SELECT cluster_id FROM ideas_raw WHERE id = %s;", (id_,))
    res = cur.fetchone()
    return res[0] if res and res[0] else id_

# ───────────────────────── MAIN ───────────────────────────────────────

def main():
    conn = mysql.connect(**MYSQL_DSN)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        embed_missing(cur)
        conn.commit()
        dedup(cur)
        conn.commit()
        print("Done – ideas_raw up-to-date.")
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
