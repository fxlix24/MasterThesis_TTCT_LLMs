"""
OpenAI implementation of LLMStore.
Requires:  pip install openai>=1.3.0
"""

import os
import openai
from PromptEngine.core_store import LLMStore
from PromptEngine.get_prompt import get_active_prompt

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIStore(LLMStore):
    _env_model = os.getenv("OPENAI_MODEL") #Used when script is run individually 

    def _call_llm(self, prompt: str, model: str | None = None):
        model_name = model or self._env_model

        response = client.responses.create(
            model=model_name,
            instructions="You are a creative assistant.",
            input=prompt,
        )
        text = response.output_text if hasattr(response, "output_text") else None
        tokens_used = getattr(response.usage, "output_tokens", None)
        return text, tokens_used


if __name__ == "__main__":
    OpenAIStore().run(get_active_prompt())
