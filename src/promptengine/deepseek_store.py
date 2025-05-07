"""
DeepSeek implementation of LLMStore.
Requires:  pip install openai
"""

import os
from openai import OpenAI
from PromptEngine.core_store import LLMStore
from PromptEngine.get_prompt import get_active_prompt


client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

class DeepSeekStore(LLMStore):
    _env_model = os.getenv("DEEPSEEK_MODEL") #Used when script is run individually 

    def _call_llm(self, prompt: str, model: str | None = None):

        model_name = model or self._env_model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        tokens_used = response.usage.completion_tokens
        return text, tokens_used


if __name__ == "__main__":
    DeepSeekStore().run(get_active_prompt())
