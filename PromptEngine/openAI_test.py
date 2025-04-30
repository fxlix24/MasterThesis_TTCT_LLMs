import os
from openai import OpenAI
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv("automation.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt = "List as many alternate uses for a brick as you can."
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a creative assistant.",
    input=prompt,
    # You can add additional parameters here (e.g., temperature, max_tokens) if supported.
)
print(response.output_text.strip() if hasattr(response, "output_text") else None)
print(getattr(response.usage, "output_tokens", None))
