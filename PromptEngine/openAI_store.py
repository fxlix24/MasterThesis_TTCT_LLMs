"""
OpenAI implementation of LLMStore.
Requires:  pip install openai>=1.3.0
"""

import os
import openai
from core_store import LLMStore
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv("automation.env")


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
        tokens_used = getattr(response.usage, "total_tokens", None)
        return text, tokens_used


if __name__ == "__main__":
    OpenAIStore().run("List as many alternate uses for a brick as you can.")
