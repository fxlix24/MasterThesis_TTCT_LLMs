# scripts/evaluate_requests.py
"""
Iterates over all Request records and stores / updates Evaluation rows.
The evaluation *instructions* are constant, the *input* is passed as JSON.
"""

import os, sys, json, re
from sqlalchemy.orm import joinedload
from datetime import datetime
from pathlib import Path
# ── project-local imports ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from promptengine.core_store import Session, Request, Evaluation          
from evaluation.llm_judge import LLMJudge 
from evaluation.build_payload import build_payload                   
# ────────────────────────────────────────────────────────────────────────


# ── 1. static evaluation set up (no request-specific info!) ──────
prompt_file = os.getenv("EVALUATION_TEXTFILE")
if prompt_file is None:
    raise RuntimeError(
        "EVALUATION_TEXTFILE is not set in the environment (.env)."
    )

prompt_path = Path(prompt_file)
if not prompt_path.is_absolute():
    prompt_path = Path(ROOT) / prompt_path   # ROOT comes from your snippet

if not prompt_path.exists():
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

EVAL_INSTRUCTIONS = prompt_path.read_text(encoding="utf-8")
judge_model = os.getenv("JUDGE_MODEL")


def main() -> None:
    sess = Session()

    total = sess.query(Request).count()
    already = sess.query(Evaluation).count()
    print(f" {total} requests found ─ {already} already evaluated.")

    if total == 0:
        print("Nothing to do – DB is empty.")
        return
    
    print("\n━━━━━━━━ Evaluation run preview ━━━━━━━━")
    print(f"Model to be used: {judge_model}")

    # show the full prompt but wrapped to 80-columns so it’s readable
    print("\nPrompt to be used:\n")
    print(EVAL_INSTRUCTIONS)
    print()

    to_eval_if_skip = total - already
    print(f"Database entries found: {total}")
    print(f"Already evaluated:      {already}")
    print(f"Would be evaluated now: {to_eval_if_skip} "
        "(or all entries if you choose to overwrite)")
    print("────────────────────────────────────────\n")

    proceed = input("Proceed with these settings? (y/N): ").strip().lower()
    if proceed != "y":
        print("Aborted by user.")
        return

    # ── 2. ask once whether to overwrite existing evaluations ───────────
    override = False
    if already:
        user_in = input(
            f"{already} evaluations already exist. Overwrite them? (y/N): "
        ).strip().lower()
        override = user_in == "y"
        if override:
            print("Existing evaluations **will be updated**.\n")

    # ── 3. init judge ───────────────────────────────────────────────────
    judge = LLMJudge(model=judge_model)

    # ── 4. process each request ─────────────────────────────────────────
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

        # ---- build input payload ---------------------------------------
        payload = build_payload(req) # Build Payload using seperate script 
        result = judge.evaluate(EVAL_INSTRUCTIONS, payload)

        if not result:
            print(f"ID {req.id:>5} – evaluation failed")
            sess.rollback()
            failed += 1
            continue

        # ---- store (insert or update) ----------------------------------
        if existing:
            existing.originality = result["originality"]
            existing.fluency     = result["fluency"]
            existing.flexibility = result["flexibility"]
            existing.elaboration = result["elaboration"]
            existing.timestamp   = datetime.utcnow()
        else:
            sess.add(
                Evaluation(
                    request_id=req.id,
                    **result,
                    timestamp=datetime.utcnow(),
                )
            )

        sess.commit()
        print(f"ID {req.id:>5} – stored ✓  {json.dumps(result)}")
        success += 1

    # ── 5. summary ──────────────────────────────────────────────────────
    print("\n── run complete ──────────────────────────────────────────────")
    print(f"processed: {processed}")
    print(f"evaluated: {success}")
    print(f"failed:    {failed}")
    print(f"skipped:   {skipped}")

    sess.close()


if __name__ == "__main__":
    main()
