# Main script that initates the data collection for the currently selected AUT Experiment determined in the automation.env file with the parameter PROJECT_PHASE
# ------- !!! WARNING THIS SCRIPT IS NOT READY FOR USE YET !!! -------
import itertools, sys, time, json, os
from pathlib import Path
from dotenv import load_dotenv
from PromptEngine.get_prompt import get_active_prompt        

from PromptEngine.openAI_store import OpenAIStore
from PromptEngine.gemini_store import GeminiStore
from PromptEngine.deepseek_store import DeepSeekStore
from PromptEngine.anthropic_store import AnthropicStore

load_dotenv(Path(__file__).with_name("automation.env"))   # credentials

# ----- 1. Define which models to test ------------------
MODEL_MATRIX: dict[str, list[str]] = {
    "openai":  ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "o3", "o4-mini"],
    "gemini":  ["gemini-2.0-flash", "gemini-2.5-pro"],
    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"],
    "deepseek":  ["deepseek-chat"],
    # … add more vendors / models here …
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

# ----- 3. Little helpers --------------------------------
VENDOR_STORE = {
    "openai":    OpenAIStore(),
    "gemini":    GeminiStore(),
    "anthropic": AnthropicStore(),
    "deepseek": DeepSeekStore(),
}

def call_model(vendor: str, model: str, prompt: str):
    store = VENDOR_STORE[vendor]
    text, tokens = store._call_llm(prompt, model=model)
    return dict(vendor=vendor, model=model, text=text, tokens=tokens)

# ----- 4. Main loop -------------------------------------
results = []
for vendor, models in MODEL_MATRIX.items():
    for model in models:
        for _ in range(RUNS_PER_MODEL):
            res = call_model(vendor, model, prompt)
            results.append(res)
            print(f"[{vendor}/{model}] tokens={res['tokens']}")

# ----- 5. Persist / post-process -------------------------
Path("out").mkdir(exist_ok=True)
ts   = time.strftime("%Y%m%d-%H%M%S")
Path(f"out/raw_{ts}.jsonl").write_text("\n".join(json.dumps(r) for r in results))
print(f"\nSaved {len(results)} rows to out/raw_{ts}.jsonl")
