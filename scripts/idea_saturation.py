# idea_saturation.py  ─────────────────────────────────────────────────────
from collections import defaultdict
from typing import Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from database import engine, Request, Response        
import spacy, nltk
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 1 ▸ NLP helpers                                                    #
# ------------------------------------------------------------------ #
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

try:
    STOP = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    try:
        nltk.download("stopwords", quiet=True, raise_on_error=True)
        STOP = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        # Fallback: spaCy-Stopwords (immer dabei)
        print("NLTK-Stopwords not available – using spaCy-Stopwords.")
        STOP = nlp.Defaults.stop_words

def canonical(text: str) -> str:
    """
    Lemma- + stop-word-normalised signature of an idea.
    Alphabetical order makes signatures deterministic.
    """
    doc = nlp(text.lower())
    toks = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOP]
    return " ".join(sorted(toks))

# ------------------------------------------------------------------ #
# 2 ▸ Pull the bullets once, do the normalisation in Python          #
# ------------------------------------------------------------------ #
with Session(engine) as s:
    rows = s.execute(
        select(
            Request.id.label("request_id"),
            Request.model.label("model"),
            Response.bullet_point.label("idea")         # already stripped!
        ).join(Response)
        .order_by(Request.id, Response.bullet_number)
    ).all()

df = (
    pd.DataFrame(rows)
      .assign(canon=lambda d: d["idea"].map(canonical))   # new column
)

# ------------------------------------------------------------------ #
# 3 ▸ Saturation logic                                               #
# ------------------------------------------------------------------ #
def plateau(sub: pd.DataFrame, window: int = 1) -> Tuple[int|None, int]:
    """
    Returns (plateau_request_id, total_unique_ideas).
    plateau_request_id is the *first* request after which `window`
    consecutive requests added **zero** new canonical ideas.
    """
    seen, stagn = set(), 0, None
    for rid, ideas in sub.groupby("request_id")["canon"]:
        fresh = any(idea not in seen for idea in ideas)
        seen.update(ideas)
        stagn = 0 if fresh else stagn + 1
        if stagn+1 >= window:           # plateau reached
            return rid, len(seen)
    return None, len(seen)              # never saturated

result = defaultdict(dict)
for model, sub in tqdm(df.groupby("model"), desc="models"):
    req, total = plateau(sub, window=1)
    result[model]["plateau_request"] = req
    result[model]["distinct_ideas"]  = total

summary = pd.DataFrame(result).T.sort_values("plateau_request")
print(summary)
