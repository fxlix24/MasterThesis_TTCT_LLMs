# Main script that initates the data collection for the currently selected AUT Experiment determined in the automation.env file with the parameter PROJECT_PHASE
import sys, os
from pathlib import Path
from dotenv import load_dotenv
from promptengine.get_prompt import get_active_prompt        

# import of storing implementations
from promptengine.openai_client import OpenAIClient
from promptengine.gemini_client import GeminiClient
from promptengine.deepseek_client import DeepSeekClient
from promptengine.anthropic_client import AnthropicClient
from promptengine.core_store import AbstractLLMStore
from promptengine.test_runner_service import TestRunnerService

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
VENDOR_CLIENT = {
    "openai":    OpenAIClient(),
    "gemini":    GeminiClient(),
    "anthropic": AnthropicClient(),
    "deepseek": DeepSeekClient(),
}

VENDOR_STORE = AbstractLLMStore();

def call_model(vendor: str, prompt: str, model: str):
    runner = TestRunnerService(
        VENDOR_CLIENT[vendor],
        VENDOR_STORE,
    )    

    return runner.run_bench(prompt, model)

# ----- 4. Main loop -------------------------------------
for vendor, models in MODEL_MATRIX.items():
    for model in models:
        for _ in range(int(RUNS_PER_MODEL)):
            res_token = call_model(vendor, prompt, model)
            print(f"[{vendor}/{model}] tokens={res_token}")
