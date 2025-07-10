#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from typing import List

import pymysql
from dotenv import load_dotenv

# ───────────────────────── DB SET-UP ──────────────────────────────
load_dotenv("automation.env")

MYSQL_DSN = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", 3306)),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "aut_experiment"),
    "charset":  "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

EXPERIMENT_PHASE = os.getenv("PROJECT_PHASE") # Alternatively use 'os.getenv("PROJECT_PHASE")' --> returns Integer of currently selected PROJECT_PHASE
REQ_TABLE      = "requests"
RESP_TABLE     = "responses"
IDEAS_TABLE    = f"ideas_aut_{EXPERIMENT_PHASE}"


# ─────────────────────── CORE FUNCTIONS ───────────────────────────
def fetch_missing_request_ids(conn) -> List[int]:
    """Return request IDs that have responses but are absent from `ideas_raw`."""
    sql = (
        f"SELECT DISTINCT r.id AS request_id "
        f"FROM   {REQ_TABLE} r "
        f"JOIN   {RESP_TABLE} resp ON resp.request_id = r.id "
        f"LEFT JOIN {IDEAS_TABLE} i ON i.request_id = r.id "
        f"WHERE  i.request_id IS NULL AND r.experiment_phase = {EXPERIMENT_PHASE}"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        return [row["request_id"] for row in cur.fetchall()]


def fetch_rows_for_requests(conn, missing_ids: List[int]):
    """Pull *(request_id, model, bullet_number, bullet_point)* for the IDs."""
    if not missing_ids:
        return []

    fmt_ids = ",".join(["%s"] * len(missing_ids))
    sql = (
        f"SELECT r.id            AS request_id, "
        f"       r.model         AS model, "
        f"       s.bullet_number AS bullet_number, "
        f"       s.bullet_point  AS bullet_point "
        f"FROM   {RESP_TABLE} s "
        f"JOIN   {REQ_TABLE}  r ON r.id = s.request_id "
        f"WHERE  r.id IN ({fmt_ids})"
    )
    with conn.cursor() as cur:
        cur.execute(sql, missing_ids)
        return cur.fetchall()


def insert_into_ideas_raw(conn, rows) -> int:
    sql = (
        f"INSERT INTO {IDEAS_TABLE} "
        f"(request_id, model, bullet_number, bullet_point) "
        f"VALUES (%s, %s, %s, %s)"
    )
    data = [
        (row["request_id"], row["model"], row["bullet_number"], row["bullet_point"])
        for row in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, data)
    conn.commit()
    return len(data)

def sync_ideas():
    try:
        conn = pymysql.connect(**MYSQL_DSN)
    except pymysql.MySQLError as e:
        print(f"[sync_ideas_raw] Could not connect to MySQL: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Check for requests that have no responses at all
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count 
                FROM {REQ_TABLE} r 
                LEFT JOIN {RESP_TABLE} resp ON resp.request_id = r.id 
                WHERE resp.request_id IS NULL
            """)
            empty_requests = cur.fetchone()['count']
            if empty_requests > 0:
                print(f"[INFO] Found {empty_requests} requests with no response data (will be skipped)")

        missing_ids = fetch_missing_request_ids(conn)
        if not missing_ids:
            print("[sync_ideas_raw] ✔︎ ideas_raw is already up-to-date.")
            return

        print(f"[DEBUG] Request IDs with responses but missing from ideas_raw: {missing_ids[:5]}...")
        
        rows = fetch_rows_for_requests(conn, missing_ids)
        print(f"[DEBUG] Found {len(rows)} rows to insert")
        
        inserted = insert_into_ideas_raw(conn, rows)
        print(f"[sync_ideas_raw] ✔︎ Inserted {inserted} rows for {len(missing_ids)} requests with responses.")
    finally:
        conn.close()

# ─────────────────────────── MAIN ─────────────────────────────────
if __name__ == "__main__":
    sync_ideas()