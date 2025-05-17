# scripts/evaluate_requests.py
"""
Iterates over all Request records and stores / updates Evaluation rows.
The evaluation *instructions* are constant, the *input* is passed as JSON.
"""

from __future__ import annotations

import os
import re
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sqlalchemy.orm import joinedload

# ── project‑local imports ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from promptengine.core_store import Session, Request, Evaluation      
from evaluation.llm_judge        import LLMJudge                     
from evaluation.build_payload    import build_payload                 
# ───────────────────────────────────────────────────────────────────────

# ── 1.  load *one prompt per dimension*  ───────────────────────────────
DIMENSIONS: List[str] = [
    "originality",
    "fluency",
    "flexibility",
    "elaboration",
]

PROMPTS_DIR = os.getenv("EVAL_PROMPTS_DIR")
if PROMPTS_DIR is None:
    raise RuntimeError(
        "EVAL_PROMPTS_DIR is not set – point it to the folder containing the"
        " four dimension‑specific prompt files (originality.txt, …)."
    )

prompt_dir_path = Path(PROMPTS_DIR).expanduser().resolve()
if not prompt_dir_path.is_dir():
    raise FileNotFoundError(f"Prompt directory not found: {prompt_dir_path}")

EVAL_PROMPTS: Dict[str, str] = {}
for dim in DIMENSIONS:
    p = prompt_dir_path / f"{dim}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing prompt file for '{dim}': {p}")
    EVAL_PROMPTS[dim] = p.read_text(encoding="utf-8")

judge_model = os.getenv("JUDGE_MODEL")

# ── 2.  helper – extract a *single* score from any judge response ——

def _extract_score(dim: str, result) -> int:
    if result is None:
        raise ValueError("judge returned None")

    # ─ dry‑run path (result is full dict) ────────────────────────────
    if isinstance(result, dict):
        if dim in result:
            return int(result[dim])
        if "score" in result:
            return int(result["score"])
        # fallback when evaluate() still returns all four dims
        if all(k in result for k in DIMENSIONS):
            return int(result[dim])

    # ─ simple scalar already ─────────────────────────────────────────
    if isinstance(result, (int, float)):
        return int(result)

    raise ValueError(
        f"Cannot extract '{dim}' score from judge response: {result!r}")


# ── 3.  main routine ───────────────────────────────────────────────────

def main() -> None:
    sess = Session()

    total    = sess.query(Request).count()
    already  = sess.query(Evaluation).count()
    to_score = total - already

    print(f"{total} requests found – {already} already evaluated.\n")
    if total == 0:
        print("Nothing to do – DB is empty.")
        return

    print("━━━━━━━━ Evaluation run preview ━━━━━━━━")
    print(f"Model to be used: {judge_model}")
    print("Dimension-specific prompts:")
    for d in DIMENSIONS:
        m = re.search(r"────────────────\s*SCORING RUBRIC\s*────────────────(.*?)(?:────────────────|$)",
                EVAL_PROMPTS[d], re.S)
        if m:
            rubric = m.group(1).strip()
            # indent each line of the rubric for readability
            for line in rubric.splitlines():
                print(f"     {line}")
        else:
            print("     [no SCORING RUBRIC found]")
    print("────────────────────────────────────────\n")

    proceed = input("Proceed with these settings? (y/N): ").strip().lower()
    if proceed != "y":
        print("Aborted by user.")
        return

    # ── ask once whether to overwrite existing evaluations ──────────
    override = False
    if already:
        override = input(
            f"{already} evaluations already exist. Overwrite them? (y/N): "
        ).strip().lower() == "y"
        if override:
            print("Existing evaluations **will be updated**.\n")

    # ── init judge client ────────────────────────────────────────────
    judge = LLMJudge(model=judge_model)

    # ── iterate over requests ────────────────────────────────────────
    processed = success = failed = skipped = 0

    requests = (
        sess.query(Request)
        .options(joinedload(Request.responses))
        .order_by(Request.id)
        .all()
    )

    for req in requests:
        processed += 1
        existing = sess.query(Evaluation).filter_by(request_id=req.id).first()

        if existing and not override:
            skipped += 1
            continue

        if not req.responses:
            print(f"ID {req.id:>5} – no responses → skipped")
            skipped += 1
            continue

        # ---- build common payload ----------------------------------
        payload = build_payload(req)
        scores: Dict[str, int] = {}

        # ---- evaluate *each* dimension sequentially -----------------
        for dim in DIMENSIONS:
            instructions = EVAL_PROMPTS[dim]
            result       = judge.evaluate(instructions, payload)
            try:
                scores[dim] = _extract_score(dim, result)
            except ValueError as err:
                print(f"ID {req.id:>5} – {dim} evaluation failed: {err}")
                sess.rollback()
                failed += 1
                break  # abandon this request entirely

        if len(scores) != 4:
            # at least one dimension failed – skip storing anything
            continue

        # ---- store (insert or update) only when *all four* present --
        now_utc = datetime.now(timezone.utc)
        if existing:
            existing.originality = scores["originality"]
            existing.fluency     = scores["fluency"]
            existing.flexibility = scores["flexibility"]
            existing.elaboration = scores["elaboration"]
            existing.timestamp   = now_utc
        else:
            sess.add(
                Evaluation(
                    request_id  = req.id,
                    originality = scores["originality"],
                    fluency     = scores["fluency"],
                    flexibility = scores["flexibility"],
                    elaboration = scores["elaboration"],
                    timestamp   = now_utc,
                )
            )

        sess.commit()
        print(f"ID {req.id:>5} – stored ✓  {json.dumps(scores)}")
        success += 1

    # ── summary ──────────────────────────────────────────────────────
    print("\n── run complete ──────────────────────────────────────────────")
    print(f"processed: {processed}")
    print(f"evaluated: {success}")
    print(f"failed:    {failed}")
    print(f"skipped:   {skipped}")

    sess.close()


if __name__ == "__main__":
    main()
