# test_runner_service.py
from promptengine.core_llm   import AbstractLLM
from promptengine.core_store import AbstractLLMStore

class TestRunnerService:
    def __init__(self, llm: AbstractLLM, store: AbstractLLMStore):
        self.llm   = llm
        self.store = store

    def run_bench(self, prompt: str, model: str) -> int:
        model_name, full_text, tokens = self.llm._call_llm(prompt, model)
        req = self.store.save(prompt, model_name, full_text, tokens)
        return req.total_tokens