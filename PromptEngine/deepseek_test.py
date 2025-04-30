import os
from openai import OpenAI
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv("automation.env")

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "List as many alternative use cases of a cup as you can."}, # Examples of alternative use cases include: draw a circle, boil things in, make sandcastles"},
    ],
    stream=False
)

print(response.choices[0].message.content)
print(response.usage.completion_tokens)