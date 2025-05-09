"""
Gemini implementation of LLMStore.
Requires:  pip install google-generativeai
"""

import os, json
import google.generativeai as genai
from promptengine.core_store import LLMStore
from promptengine.get_prompt import get_active_prompt


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiStore(LLMStore):
    _env_model = os.getenv("GOOGLE_MODEL") #Used when script is run individually 

    def _call_llm(self, prompt: str, model: str | None = None):
        model_name = model if model is not None else self._env_model

        model = genai.GenerativeModel(model_name)
        resp  = model.generate_content("You are a creative assistant. " +
            prompt
        )
        text = resp.text if resp.text else None
        tokens_used = resp.usage_metadata.total_token_count - resp.usage_metadata.prompt_token_count if resp.usage_metadata else None
        return model_name, text, tokens_used


if __name__ == "__main__":
    GeminiStore().run(get_active_prompt())
