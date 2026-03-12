"""
API Connection Test
===================
This test verifies the OpenRouter API connection:
- API key validation
- Model configuration
- Basic completion request
- Reasoning capability (if supported by model)

Usage:
    python test_api.py
"""
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "CRITICAL ERROR: OPENROUTER_API_KEY not found. Check your .env file."
    )

logging.info("API Key loaded securely. Initializing client...")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

logging.info("Making test API call to OpenRouter...")
try:
    response = client.chat.completions.create(
        model="arcee-ai/trinity-large-preview:free",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: 'API connection verified.'",
            }
        ],
        max_tokens=20,
        extra_body={"reasoning": {"enabled": True}},
    )

    content = response.choices[0].message.content.strip()
    logging.info(f"✅ API Response: {content}")

    reasoning = getattr(response.choices[0].message, "reasoning", None)
    if reasoning:
        logging.info(f"✅ Reasoning captured: {len(reasoning)} chars")
    else:
        logging.info(
            "ℹ️  No separate reasoning field returned (model may embed it in content)"
        )

    logging.info("✅ API SMOKE TEST PASSED!")

except Exception as e:
    logging.error(f"❌ API call failed: {e}")
    raise SystemExit(1)
