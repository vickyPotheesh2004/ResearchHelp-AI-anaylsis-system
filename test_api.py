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
models_to_test = [
    ("GLM 4.5 Air (Titler/Mermaid)", "z-ai/glm-4.5-air:free", False),
    ("Gemma 3 12B (Standard)", "google/gemma-3-12b-it:free", False),
    ("Trinity Large (QA Reasoning)", "arcee-ai/trinity-large-preview:free", True),
    ("Nemotron 3 Super (Research Reasoning)", "nvidia/nemotron-3-super-120b-a12b:free", True),
]

for name, model_id, enable_reasoning in models_to_test:
    logging.info(f"Testing {name} [{model_id}]...")
    try:
        extra_body = {"reasoning": {"enabled": True}} if enable_reasoning else None
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly 'OK'",
                }
            ],
            max_tokens=10,
            extra_body=extra_body,
        )

        content = response.choices[0].message.content.strip()
        logging.info(f"✅ {name} Response: {content}")
        
        if enable_reasoning:
            # Check for reasoning field in different possible locations
            reasoning = getattr(response.choices[0].message, "reasoning", None)
            if not reasoning and hasattr(response.choices[0].message, "reasoning_details"):
                reasoning = response.choices[0].message.reasoning_details
            
            if reasoning:
                logging.info(f"✅ Reasoning captured for {name}")
            else:
                logging.warning(f"⚠️ No separate reasoning field for {name}")

    except Exception as e:
        logging.error(f"❌ {name} failed: {e}")

logging.info("✅ API CONNECTIVITY TEST COMPLETE!")
