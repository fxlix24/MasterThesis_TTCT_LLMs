"""
Anthropic implementation of LLMStore.
Requires:  pip install anthropic
"""

import os
import anthropic
from promptengine.core_store import LLMStore
from promptengine.get_prompt import get_active_prompt


client = anthropic.Anthropic()

class AnthropicStore(LLMStore):
    _env_model = os.getenv("ANTHROPIC_MODEL") #Used when script is run individually 

    def _call_llm(self, prompt, model: str | None = None):
        model_name = model if model is not None else self._env_model

        resp = client.messages.create(
            model=model_name,
            max_tokens=1000,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        # resp.content is a list[TextBlock]; join the text blocks
        text = "\n".join(block.text for block in resp.content if block.type == "text")
        return model_name, text, resp.usage.output_tokens

if __name__ == "__main__":
    AnthropicStore().run(get_active_prompt())
