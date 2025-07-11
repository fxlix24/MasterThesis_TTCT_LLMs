# scripts/calculate_fluency.py
"""
Iterates over all Request records, counts the number of associated responses,
and stores this count as the fluency score in the Evaluation table.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

# ── project‑local imports ──────────────────────────────────────────────
# Stellt sicher, dass die Datenbankmodule korrekt geladen werden.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from database import engine, Request, Evaluation
# ───────────────────────────────────────────────────────────────────────


def calculate_and_store_fluency():
    """
    Stellt eine Verbindung zur Datenbank her, um die Fluency-Werte zu berechnen
    und zu speichern.
    """
    with Session(engine) as sess:
        try:
            print("Starte die Berechnung der Fluency-Werte...")

            # Ruft alle Anfragen mit den zugehörigen Antworten ab.
            # `joinedload` optimiert die Abfrage, indem es die Antworten
            # effizient mitlädt.
            requests = sess.query(Request).options(joinedload(Request.responses)).all()

            if not requests:
                print("Keine Anfragen in der Datenbank gefunden.")
                return

            print(f"{len(requests)} Anfragen gefunden. Verarbeite jetzt...")

            updated_count = 0
            created_count = 0

            # Iteriert über jede Anfrage, um die Anzahl der Antworten zu zählen.
            for req in requests:
                fluency_score = len(req.responses)

                # Prüft, ob bereits ein Auswertungseintrag für diese Anfrage existiert.
                evaluation = sess.query(Evaluation).filter_by(request_id=req.id).first()

                now_utc = datetime.now(timezone.utc)

                if evaluation:
                    # Aktualisiert einen vorhandenen Eintrag.
                    evaluation.fluency = fluency_score
                    evaluation.timestamp = now_utc
                    updated_count += 1
                    print(f"ID {req.id:>5}: Vorhandener Eintrag aktualisiert. Fluency = {fluency_score}")
                else:
                    # Erstellt einen neuen Eintrag, falls keiner existiert.
                    # Andere Bewertungsdimensionen werden auf None gesetzt.
                    new_evaluation = Evaluation(
                        request_id=req.id,
                        fluency=fluency_score,
                        originality=None,  # Oder ein Standardwert wie 0
                        flexibility=None,  # Oder ein Standardwert wie 0
                        elaboration=None,  # Oder ein Standardwert wie 0
                        timestamp=now_utc,
                    )
                    sess.add(new_evaluation)
                    created_count += 1
                    print(f"ID {req.id:>5}: Neuer Eintrag erstellt. Fluency = {fluency_score}")

            # Überträgt alle Änderungen in die Datenbank.
            sess.commit()
            print("\n── Berechnung abgeschlossen ──────────────────────────────────")
            print(f"Erfolgreich {created_count} neue Einträge erstellt.")
            print(f"Erfolgreich {updated_count} vorhandene Einträge aktualisiert.")

        except Exception as e:
            # Macht Änderungen rückgängig, falls ein Fehler auftritt.
            sess.rollback()
            print(f"\nEin Fehler ist aufgetreten: {e}")
            print("Alle Änderungen wurden rückgängig gemacht.")
        finally:
            # Die Sitzung wird durch den 'with'-Block automatisch geschlossen.
            print("Datenbankverbindung geschlossen.")


if __name__ == "__main__":
    calculate_and_store_fluency()