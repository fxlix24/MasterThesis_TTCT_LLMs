#!/usr/bin/env python3
"""
migrate_bullets.py
──────────────────
Copy the legacy `bullet_text` into the new split-columns
`bullet_point` and `bullet_explantions`.

• Uses the same SQLAlchemy engine/settings as the rest of the project
• Never overwrites data that is already present
• Works in batches so the whole job commits (or rolls back) atomically
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ------------------------------------------------------------------ #
# 0. configuration – reuse the existing DATABASE URL                 #
# ------------------------------------------------------------------ #
load_dotenv("automation.env")              # adjust if different
MYSQL_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST', 'localhost')}:{os.getenv('MYSQL_PORT', 3306)}"
    f"/{os.getenv('MYSQL_DB')}"
)

# ------------------------------------------------------------------ #
# 1. helper to split idea vs. explanation                            #
# ------------------------------------------------------------------ #

_delim = re.compile(
    r"""
    (                      
        \s[-–—]\s          
      | :\s+               
      | :\*\*\s+            
    )
    """,
    re.VERBOSE,
)

def _clean_response(text: str) -> Optional[Tuple[str, str]]:
    txt = (text or "").strip()
    if not txt:
        return None

    m = _delim.search(txt)
    if not m:                        # no delimiter found → whole line is the idea
        return (txt.rstrip(". "), "")

    head = txt[:m.start()].rstrip(". ")
    tail = txt[m.end():].lstrip().rstrip(". ")

    # If the delimiter was ":** " we snipped the closing **.
    # Put it back and strip it from the detail if necessary.
    if m.group().startswith(":**"):
        head = head + "**"
        if tail.startswith("**"):
            tail = tail[2:].lstrip()

    # final tidy-up
    return (head.strip(), tail.strip())
# ------------------------------------------------------------------ #
# 2. main routine                                                    #
# ------------------------------------------------------------------ #
def migrate(engine: Engine) -> None:
    with engine.begin() as conn:                         # single TX
        rows = conn.execute(
            text(
                """
                SELECT id, bullet_text
                FROM responses
                """
            )
        ).fetchall()

        if not rows:
            print("Nothing to migrate – new columns already populated.")
            return

        updates = []
        for _id, bullet_text in rows:
            cleaned = _clean_response(bullet_text)
            if cleaned is None:
                # keep NULLs, we only fill when there is useful text
                continue

            point, detail = cleaned
            updates.append({"id": _id, "point": point, "detail": detail})

        if not updates:
            print("No rows contained parsable bullets – aborting.")
            return

        conn.execute(
            text(
                """
                UPDATE responses
                   SET bullet_point      = :point,
                       bullet_details = :detail
                 WHERE id = :id
                """
            ),
            updates,                              # executemany()
        )
        print(f"Updated {len(updates):,} rows successfully.")


if __name__ == "__main__":
    engine = create_engine(MYSQL_URL, pool_pre_ping=True, future=True)
    migrate(engine)
