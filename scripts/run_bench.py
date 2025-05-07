# Main script that initates the data collection for the currently selected AUT Experiment determined in the automation.env file with the parameter PROJECT_PHASE
import itertools, sys, time, json, os
from pathlib import Path
from dotenv import load_dotenv
from promptengine.get_prompt import get_active_prompt        

# import of storing implementations
from promptengine.openai_store import OpenAIStore
from promptengine.gemini_store import GeminiStore
from promptengine.deepseek_store import DeepSeekStore
from promptengine.anthropic_store import AnthropicStore

load_dotenv(Path(__file__).with_name("automation.env"))   # credentials

# ----- 1. Define which models to test ------------------
MODEL_MATRIX: dict[str, list[str]] = {
    "openai":  ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "o3", "o1", "o4-mini"],
    "gemini":  ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06", "gemini-1.5-pro"],
    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"],
    "deepseek":  ["deepseek-chat", "deepseek-reasoner"],
}
RUNS_PER_MODEL = os.getenv("RUNS_PER_MODEL")

# ----- 2. Tell user what will happen -------------------
prompt = get_active_prompt()
print("\n=== BENCHMARK PLAN =================================")
print(f"Prompt ({len(prompt)} chars):\n\"{prompt}\"\n")
print("Models × runs:")
for vendor, models in MODEL_MATRIX.items():
    for m in models:
        print(f"  {vendor:10s}  {m:35s}  × {RUNS_PER_MODEL}")
print("=====================================================")
go = input("Proceed? [y/N] ").strip().lower()
if go != "y":
    sys.exit("Aborted.\n")

# ----- 3. Initialize Storing --------------------------------
VENDOR_STORE = {
    "openai":    OpenAIStore(),
    "gemini":    GeminiStore(),
    "anthropic": AnthropicStore(),
    "deepseek": DeepSeekStore(),
}

def call_model(vendor: str, prompt: str, model: str):
    store = VENDOR_STORE[vendor]
    req = store.run(prompt, model=model)
    text   = "\n".join(r.bullet_text for r in req.responses)
    tokens = req.total_tokens

    return {
        "db_id":   req.id,
        "vendor":  vendor,
        "model":   req.model,
        "tokens":  tokens,
        "text":    text,
    }

# ----- 4. Main loop -------------------------------------
results = []
for vendor, models in MODEL_MATRIX.items():
    for model in models:
        for _ in range(int(RUNS_PER_MODEL)):
            res = call_model(vendor, prompt, model)
            results.append(res)
            print(f"[{vendor}/{model}] tokens={res['tokens']}")
