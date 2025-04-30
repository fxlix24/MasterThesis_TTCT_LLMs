"""
Anthropic implementation of LLMStore.
Requires:  pip install anthropic
"""

import os
import anthropic
from core_store import LLMStore
from get_prompt import get_active_prompt


client = anthropic.Anthropic()

class AnthropicStore(LLMStore):
    model_name = os.getenv("ANTHROPIC_MODEL")

    def _call_llm(self, prompt):
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        # resp.content is a list[TextBlock]; join the text blocks
        text = "\n".join(block.text for block in resp.content if block.type == "text")
        return text, resp.usage.output_tokens

if __name__ == "__main__":
    AnthropicStore().run(get_active_prompt())
