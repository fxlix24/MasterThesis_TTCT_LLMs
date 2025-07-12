# src/evaluation/llm_judge.py
import json, os, random, openai
from typing import Any, Dict, Optional

class LLMJudge:

    def __init__(self, model) -> None:
        self.model   = model if model is not None else "placeholder-judge"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client  = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        print(f"LLMJudge using model: {self.model}")

    def _extract_json_from_markdown(self, content: str) -> dict:
    # Remove the Markdown code block fences
        if content.strip().startswith("```json"):
            # Strip leading/trailing whitespace and the markdown fences
            json_string = content.strip().removeprefix("```json\n").removesuffix("\n```")
            return json.loads(json_string)
        # Handle the case where there's no markdown block
        return json.loads(content)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        instructions: str,
        input_payload: Dict[str, Any],
    ) -> Optional[Dict[str, int]]:
        """
        Returns a dict with the four rubric items or None on error.
        """
        # ------------------------------------------------------------------ #
        # 1. Dry-run mode (no key or dummy model)                            #
        # ------------------------------------------------------------------ #
        if not self.client or self.model == "placeholder-judge":
            print("No API key or placeholder model – returning simulated scores.")
            return {
                "originality": random.randint(1, 5),
                "flexibility": random.randint(1, 5),
                "elaboration": random.randint(1, 5),
            }

        # ------------------------------------------------------------------ #
        # 2. Real call                                                       #
        # ------------------------------------------------------------------ #
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions.strip(),
                input=json.dumps(input_payload, ensure_ascii=False),
                temperature=0.0,
            )
        
            content = response.output_text if hasattr(response, "output_text") else None
            return self._extract_json_from_markdown(content)
        
        except Exception as err:
            print(f"[LLMJudge] OpenAI call failed: {err}")
            return None
        