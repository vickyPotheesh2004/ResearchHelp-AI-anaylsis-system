import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Load the hidden .env file
load_dotenv()

# 2. Fetch the key securely from memory, NOT the code
api_key = os.getenv("OPENROUTER_API_KEY")

# 3. Fail loudly if the key is missing (Good engineering practice)
if not api_key:
    raise ValueError("CRITICAL ERROR: OPENROUTER_API_KEY not found. Check your .env file.")

logging.info("API Key loaded securely. Initializing client...")

# 4. Initialize the client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# --- Your existing test logic goes below here ---
# Example:
# response = client.chat.completions.create(...)
# print(response)