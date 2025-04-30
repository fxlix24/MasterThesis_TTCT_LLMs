"""
Gemini implementation of LLMStore.
Requires:  pip install google-generativeai
"""

import os, json
import google.generativeai as genai
from core_store import LLMStore


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiStore(LLMStore):
    model_name = os.getenv("GOOGLE_MODEL")

    def _call_llm(self, prompt: str):
        model = genai.GenerativeModel(self.model_name)
        resp  = model.generate_content("You are a creative assistant. " +
            prompt
        )
        text = resp.text if resp.text else None
        tokens_used = resp.usage_metadata.total_token_count if resp.usage_metadata else None
        return text, tokens_used


if __name__ == "__main__":
    GeminiStore().run("List as many alternate uses for a brick as you can.")
