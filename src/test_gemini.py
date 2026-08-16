import os

from dotenv import load_dotenv
from google import genai


# Load variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


print("API key found.")

client = genai.Client(
    api_key=api_key
)


response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain artificial intelligence in 2 simple sentences."
)


print("\nGemini response:")
print(response.text)