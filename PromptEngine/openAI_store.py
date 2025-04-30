"""
OpenAI implementation of LLMStore.
Requires:  pip install openai>=1.3.0
"""

import os
import openai
from core_store import LLMStore
from get_prompt import get_active_prompt

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIStore(LLMStore):
    model_name = os.getenv("OPENAI_MODEL")

    def _call_llm(self, prompt: str):
        response = client.responses.create(
            model=self.model_name,
            instructions="You are a creative assistant.",
            input=prompt,
        )
        text = response.output_text if hasattr(response, "output_text") else None
        tokens_used = getattr(response.usage, "output_tokens", None)
        return text, tokens_used


if __name__ == "__main__":
    OpenAIStore().run(get_active_prompt())
