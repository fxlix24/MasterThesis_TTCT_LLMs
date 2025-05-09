import os
import anthropic

from dotenv import load_dotenv
# Load variables from .env file
load_dotenv("automation.env")

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
message = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1000,
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List three alternate usescases of a brick"
                }
            ]
        }
    ]
)
print(message.content)
print("Total Tokens: ", message.usage.output_tokens)
