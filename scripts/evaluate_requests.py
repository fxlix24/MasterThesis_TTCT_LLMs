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

from sqlalchemy.orm import joinedload, Session

# ── project-local imports ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from database import engine, Request, Evaluation    
from evaluation.llm_judge        import LLMJudge                     
from evaluation.build_payload    import build_payload                 
# ───────────────────────────────────────────────────────────────────────

# ── 1.  load *one prompt per dimension*  ───────────────────────────────
DIMENSIONS: List[str] = [
    "originality",
    "flexibility",
    "elaboration",
]

# ── automatically evaluated dimensions helper ─────────────────────────
REQUIRED_DIMS = set(DIMENSIONS)

def _has_all_auto_scores(ev) -> bool:
    """Return **True** if *ev* already contains a non-null score for *every*
    automatically-evaluated dimension (i.e. each item currently listed in
    **DIMENSIONS**).  This allows us to ignore rows that only contain the
    manually-scored *fluency* dimension while still treating *incomplete*
    rows as needing evaluation.
    """
    return all(getattr(ev, d) is not None for d in REQUIRED_DIMS)

PROMPTS_DIR = os.getenv("EVAL_PROMPTS_DIR")
if PROMPTS_DIR is None:
    raise RuntimeError(
        "EVAL_PROMPTS_DIR is not set – point it to the folder containing the"
        " four dimension-specific prompt files (originality.txt, …)."
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

# ── 2.  helper – extract score from judge response ─────────────────────

def _extract_score(dim: str, result) -> int:
    if result is None:
        raise ValueError("judge returned None")

    # ─ dict coming straight from the judge ───────────────
    if isinstance(result, dict):
        if dim in result:            # {"originality": 3}
            return int(result[dim])
        if "score" in result:        # {"score": 3}
            return int(result["score"])
        # still allow the “all-four-dims” format
        if all(k in result for k in DIMENSIONS):
            return int(result[dim])

    # ─ simple scalar already ─────────────────────────────
    if isinstance(result, (int, float)):
        return int(result)

    raise ValueError(
        f"Cannot extract '{dim}' score from judge response: {result!r}")

# ── 3.  helper – print structure of evaluation process ─────────────────

def _printPreview(sess) -> int:

    total   = sess.query(Request).count()
    already = sess.query(Evaluation).filter(
        Evaluation.originality != None,
        Evaluation.flexibility != None,
        Evaluation.elaboration != None,
    ).count()
    to_score = total - already

    print(f"{total} requests found – {already} already fully evaluated.\n")
    if total == 0:
        print("Nothing to do – DB is empty.")
        return 0

    print("━━━━━━━━ Evaluation run preview ━━━━━━━━")
    print(f"Model to be used: {judge_model}")
    print("Dimension-specific prompts:")
    for d in DIMENSIONS:
        print(f"\n     >> Dimension: {d.title()} <<")

        m = re.search(r"╭─ SCORING .*?╮\n(.*?)\n╰",
                EVAL_PROMPTS[d], re.S)
        if m:
            rubric = m.group(1).strip()
            for line in rubric.splitlines():
                cleaned_line = line.strip().strip('│').strip()
                print(f"        {cleaned_line}")
        else:
            print(f"        [no SCORING RUBRIC found for '{d}']")

    print("\n────────────────────────────────────────\n")

    proceed = input("Proceed with these settings? (y/N): ").strip().lower()
    if proceed != "y":
        print("Aborted by user.")
        return 0

    return to_score

# ── 4.  helper – evaluation loop ───────────────────────────────────────

def _evaluateDatabase(sess, judge, override: bool):
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

        # Skip *only* if all automatically-scored dimensions are already present
        if existing and _has_all_auto_scores(existing) and not override:
            skipped += 1
            continue

        if not req.responses:
            print(f"ID {req.id:>5} – no responses → skipped")
            skipped += 1
            continue

        # ── build common payload ──────────────────────────────────────
        payload = build_payload(req)
        scores: Dict[str, int] = {}

        # ── evaluate *each* dimension sequentially ────────────────────
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

        if len(scores) != len(DIMENSIONS):
            # at least one dimension failed – skip storing anything
            continue

        if _storeResult(req, sess, existing, scores):
            success += 1
            
    return processed, success, failed, skipped

# ── 5.  helper – store results in database  ────────────────────────────

def _storeResult(req, sess, existing, scores) -> bool:

    now_utc = datetime.now(timezone.utc)

    # Enforce that all automatically-scored keys are present:
    if not REQUIRED_DIMS.issubset(scores):
            print(f"ID {req.id:>5} – missing scores, skipping store")
            return False

    if existing:
        existing.originality = scores["originality"]
        existing.flexibility = scores["flexibility"]
        existing.elaboration = scores["elaboration"]
        existing.timestamp   = now_utc
    else:
        sess.add(
            Evaluation(
                request_id  = req.id,
                originality = scores["originality"],
                flexibility = scores["flexibility"],
                elaboration = scores["elaboration"],
                timestamp   = now_utc,
            )
        )

    try:
        sess.commit()
    except Exception as e:
        sess.rollback()
        print(f"ID {req.id:>5} – store failed: {e}")
        return False

    print(f"ID {req.id:>5} – stored ✓  {json.dumps(scores)}")
    return True

# ── 6.  main routine ───────────────────────────────────────────────────

def main() -> None:
    # ── print evaluation preview ──────────────────────────────────────
    sess = Session(engine)
    total   = sess.query(Request).count()
    to_score = _printPreview(sess)
    override = to_score and input(
        "Found existing evaluations. Re-evaluate anyway? (y/N): ").strip().lower() == "y"
    sess.close()

    if to_score < total or override:
        judge = LLMJudge(judge_model)
        sess  = Session(engine)
        processed, success, failed, skipped = _evaluateDatabase(sess, judge, override)
        sess.close()

        print("\n━━━━━━━━ Summary ━━━━━━━━")
        print(f"Processed: {processed}")
        print(f"Success:   {success}")
        print(f"Failed:    {failed}")
        print(f"Skipped:   {skipped}")
    else:
        print("Nothing to do – everything already fully evaluated.")

if __name__ == "__main__":
    main()
