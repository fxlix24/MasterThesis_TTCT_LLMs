"""
DeepSeek implementation of LLMStore.
Requires:  pip install openai
"""

import os
from openai import OpenAI
from core_store import LLMStore
from get_prompt import get_active_prompt


client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

class DeepSeekStore(LLMStore):
    model_name = os.getenv("DEEPSEEK_MODEL")

    def _call_llm(self, prompt: str):
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        tokens_used = response.usage.completion_tokens
        return text, tokens_used


if __name__ == "__main__":
    DeepSeekStore().run(get_active_prompt())
