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
import os
from datetime import datetime, timezone

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
SessionLocal = sessionmaker(bind=engine)
Base    = declarative_base()

# --------------------------------------------------------------------- #
# 2. ORM tables                                                         #
# --------------------------------------------------------------------- #
class Request(Base):
    __tablename__ = "requests"

    id              = Column(Integer, primary_key=True)
    prompt          = Column(Text, nullable=False)
    timestamp       = Column(DateTime, default=datetime.now(timezone.utc))
    model           = Column(String(255), nullable=False)
    experiment_phase= Column(Integer, default=0)
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
    bullet_point  = Column(Text, nullable=False)
    bullet_details   = Column(Text, nullable=True)
    timestamp     = Column(DateTime, default=datetime.now(timezone.utc))

class Evaluation(Base):
    __tablename__ = "evaluations"

    id           = Column(Integer, primary_key=True)
    request_id   = Column(Integer, ForeignKey("requests.id"), nullable=False, unique=True)

    originality  = Column(Integer)
    fluency      = Column(Integer)
    flexibility  = Column(Integer)
    elaboration  = Column(Integer)

    timestamp    = Column(DateTime, default=datetime.now(timezone.utc))

# Create tables (harmless if they already exist)
Base.metadata.create_all(engine)