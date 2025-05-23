"""
DeepSeek implementation of LLMStore.
Requires:  pip install openai
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from promptengine.core_llm import AbstractLLM
from promptengine.get_prompt import get_active_prompt

load_dotenv("automation.env")

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

class DeepSeekClient(AbstractLLM):
    _env_model = os.getenv("DEEPSEEK_MODEL") #Used when script is run individually 

    def _call_llm(self, prompt: str, model: str | None = None):
        model_name = model if model is not None else self._env_model
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        tokens_used = response.usage.completion_tokens
        return model_name, text, tokens_used


if __name__ == "__main__":
    print(DeepSeekClient()._call_llm(get_active_prompt()))
