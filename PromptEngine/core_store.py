# core_store.py
"""
Shared database schema + reusable workflow for every model provider.
Import this file from all your *_store.py scripts.

• One SQLAlchemy Base / engine / Session for the whole project
• ORM tables: Request, Response, Evaluation
• extract_bullets() helper (newline-safe)
• Abstract class LLMStore – child scripts implement only _call_llm()
"""
from __future__ import annotations
import os, datetime, re, abc
from typing import List, Tuple

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# --------------------------------------------------------------------- #
# 1. global metadata + connection                                      #
# --------------------------------------------------------------------- #
load_dotenv("automation.env")        # adjust if your .env lives elsewhere

MYSQL_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT','3306')}"
    f"/{os.getenv('MYSQL_DB')}"
)

engine  = create_engine(MYSQL_URL, echo=False, future=True)
Session = sessionmaker(bind=engine)
Base    = declarative_base()

# --------------------------------------------------------------------- #
# 2. ORM tables                                                         #
# --------------------------------------------------------------------- #
class Request(Base):
    __tablename__ = "requests"

    id              = Column(Integer, primary_key=True)
    prompt          = Column(Text, nullable=False)
    timestamp       = Column(DateTime, default=datetime.datetime.utcnow)
    model           = Column(String(255), nullable=False)
    experiment_phase= Column(String(255), default="n/a")
    total_tokens    = Column(Integer)

    responses  = relationship("Response",  backref="request",
                               cascade="all, delete-orphan", lazy="joined")
    evaluation = relationship("Evaluation", uselist=False, backref="request")

class Response(Base):
    __tablename__ = "responses"

    id            = Column(Integer, primary_key=True)
    request_id    = Column(Integer, ForeignKey("requests.id"), nullable=False)
    bullet_number = Column(Integer, nullable=False)
    bullet_text   = Column(Text, nullable=False)
    timestamp     = Column(DateTime, default=datetime.datetime.utcnow)

class Evaluation(Base):
    __tablename__ = "evaluations"

    id           = Column(Integer, primary_key=True)
    request_id   = Column(Integer, ForeignKey("requests.id"), nullable=False, unique=True)

    originality  = Column(Integer)
    fluency      = Column(Integer)
    flexibility  = Column(Integer)
    elaboration  = Column(Integer)

    timestamp    = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables (harmless if they already exist)
Base.metadata.create_all(engine)

# --------------------------------------------------------------------- #
# 3. helper: extract numbered bullets – keeps #1 even at start of text  #
# --------------------------------------------------------------------- #
_BULLET_RE = re.compile(
    r'(?:^|\n)\s*(?:\d+\.\s*|[*\-•]\s+)(.*?)\s*(?=\n\s*(?:\d+\.|[*\-•])|\Z)',
    re.DOTALL,
)    

def extract_bullets(text: str):
    bullets = [m.group(1).strip() for m in _BULLET_RE.finditer(text)]
    return list(enumerate(bullets, start=1))
# --------------------------------------------------------------------- #
# 4. Abstract workflow – subclass per vendor                            #
# --------------------------------------------------------------------- #
class LLMStore(abc.ABC):
    """Generic “call-LLM → store prompt & ideas” workflow."""

    model_name: str = "<override in child>"

    def __init__(self, phase: str = os.getenv("PROJECT_PHASE")):
        self.phase    = phase
        self.session  = Session()

    # child implements this ------------------------------------------------
    @abc.abstractmethod
    def _call_llm(self, prompt: str) -> Tuple[str, int]:
        """Return (full_text_response, total_tokens_used)."""
        ...

    # public API -----------------------------------------------------------
    def run(self, prompt: str) -> Request:
        full_text, used_tokens = self._call_llm(prompt)

        # store request
        req = Request(
            prompt=prompt,
            model=self.model_name,
            experiment_phase=self.phase,
            total_tokens=used_tokens,
        )
        self.session.add(req)
        self.session.flush()          # populate req.id

        # split & store ideas
        for num, idea in extract_bullets(full_text):
            self.session.add(Response(
                request_id=req.id,
                bullet_number=num,
                bullet_text=idea,
            ))

        self.session.commit()
        print(f"Stored request #{req.id} with {len(req.responses)} ideas.")
        return req
