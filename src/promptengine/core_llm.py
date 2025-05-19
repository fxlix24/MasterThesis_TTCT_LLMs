# core_llm.py
import abc
from typing import Tuple

class AbstractLLM(abc.ABC):
    """Performs request on LLM Model"""
    @abc.abstractmethod
    def _call_llm(self, prompt: str, model: str | None = None) -> Tuple[str, str, int]:
        """
        → (model_name, full_text_response, total_tokens_used)
        """
        ...