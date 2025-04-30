import os
import google.generativeai as genai

from dotenv import load_dotenv
# Load variables from .env file
load_dotenv("automation.env")

# Configure the API key (replace with your API key)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create an instance of a Gemini model; here, we're using the 'gemini-1.5-flash' model.
model = genai.GenerativeModel("gemini-2.0-flash")

# Send a text prompt to the model
response = model.generate_content("List three of the most excentric things one could buy.")

# Print the generated response text
print(response.text)
print(response.usage_metadata.candidates_token_count)