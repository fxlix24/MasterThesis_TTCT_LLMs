# Loads the prompt according to the set PROJECT_PHASE in automation.env

import os
from pathlib import Path
from dotenv import load_dotenv

# Load env file relative to this script, no matter where you run Python from
load_dotenv(Path(__file__).with_name("automation.env"))

def get_active_prompt() -> str:
    """
    Return the prompt that corresponds to the current PROJECT_PHASE.
    Raises KeyError if the mapping is missing.
    """
    phase = "AUT_" + os.getenv("PROJECT_PHASE")
    if not phase:
        raise KeyError("PROJECT_PHASE is not set")

    key = f"{phase}_PROMPT"           # -> 'AUT_1_PROMPT'
    prompt = os.getenv(key)
    if not prompt:
        raise KeyError(f"{key} is not defined in automation.env")

    return prompt
