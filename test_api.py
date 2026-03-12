"""
API Connection Test
===================
This test verifies the OpenRouter API connection:
- API key validation
- Model configuration
- Basic completion request
- Reasoning capability (if supported by model)
- Vision/image understanding (Gemma 3 12B)

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
site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com")
site_title = os.getenv("OPENROUTER_SITE_TITLE", "ResearchHelp AI Analysis System")

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

# Define models with their configuration
# GLM 4.5 Air - Fast, good for mermaid code generation
# Gemma 3 12B - Multimodal, good for image understanding
# Trinity Large - Reasoning capable
# Nemotron 3 Super - Best reasoning for complex tasks

models_to_test = [
    ("GLM 4.5 Air (Mermaid/Fast)", "z-ai/glm-4.5-air:free", False, None),
    ("Gemma 3 12B (Standard)", "google/gemma-3-12b-it:free", False, None),
    ("Trinity Large (QA Reasoning)", "arcee-ai/trinity-large-preview:free", True, None),
    ("Nemotron 3 Super (Research Reasoning)", "nvidia/nemotron-3-super-120b-a12b:free", True, None),
]

extra_headers = {
    "HTTP-Referer": site_url,
    "X-OpenRouter-Title": site_title,
}

for name, model_id, enable_reasoning, image_url in models_to_test:
    logging.info(f"Testing {name} [{model_id}]...")
    try:
        extra_body = {"reasoning": {"enabled": True}} if enable_reasoning else {}
        
        # Prepare messages based on whether it's a vision test
        if image_url:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": "Reply with exactly 'OK'",
                }
            ]
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=20 if not image_url else 500,
            temperature=0.3,
            extra_headers=extra_headers,
            extra_body=extra_body if extra_body else None,
        )

        message = response.choices[0].message
        content = message.content
        
        if content:
            logging.info(f"✅ {name} Response Content: {content.strip()}")
        else:
            logging.warning(f"⚠️ {name} returned NULL content.")
            # Check for reasoning if content is null
            reasoning = getattr(message, "reasoning", None)
            if not reasoning and hasattr(message, "reasoning_details"):
                reasoning = message.reasoning_details
            
            if reasoning:
                logging.info(f"💡 {name} returned REASONING only: {str(reasoning)[:100]}...")
            else:
                logging.error(f"❌ {name} returned NO content and NO reasoning. Full message: {message}")

        if enable_reasoning and not content:
             logging.info(f"Note: Some reasoning models may suppress 'content' for simple prompts.")

    except Exception as e:
        logging.error(f"❌ {name} failed: {e}")
        if "404" in str(e):
            logging.error("   TIP: Check your OpenRouter Privacy Settings (Data Retention / Non-free models)")

# Test Vision (Gemma 3 12B) with an example image
logging.info("\n" + "="*50)
logging.info("Testing Vision Capability with Gemma 3 12B...")
logging.info("="*50)

try:
    # Example image from the user's spec
    vision_model = "google/gemma-3-12b-it:free"
    test_image_url = "https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg"
    
    response = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": test_image_url}}
                ]
            }
        ],
        max_tokens=500,
        temperature=0.3,
        extra_headers=extra_headers,
    )
    
    content = response.choices[0].message.content
    logging.info(f"✅ Vision Response: {content[:200]}...")
    
except Exception as e:
    logging.error(f"❌ Vision test failed: {e}")

logging.info("\n✅ API CONNECTIVITY TEST COMPLETE!")
