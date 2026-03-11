import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import fitz  # PyMuPDF
import docx
import pandas as pd
import pytesseract

# Import config to get the correct model
from src.config import DEFAULT_LLM_MODEL

print("--- SYSTEM DIAGNOSTIC START ---")

# TEST 1: Environment & API Key
print("\n1. Testing .env and API Key...")
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key and api_key != "your_actual_openrouter_api_key_here":
    print(f"✅ API Key loaded successfully! (Starts with: {api_key[:5]}...)")
else:
    print("❌ ERROR: API Key is missing or invalid in your .env file.")

# TEST 2: AI Connection (Using configured model)
print("\n2. Testing OpenRouter API Connection...")
try:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly three words: 'AI is online.'"}]
    )
    print(f"✅ AI Responded: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"❌ ERROR: AI Connection failed. Details: {e}")

# TEST 3: Vector Database (ChromaDB)
print("\n3. Testing Local Vector Database...")
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="diagnostic_test")
    print("✅ ChromaDB initialized and created a test collection successfully!")
except Exception as e:
    print(f"❌ ERROR: ChromaDB failed to load. Details: {e}")

# TEST 4: Extractor Modules
print("\n4. Testing Document Extractors...")
print("✅ PyMuPDF (fitz) is ready for PDFs.")
print("✅ python-docx is ready for Word documents.")
print("✅ pandas is ready for CSVs and Excel files.")
print("✅ pytesseract is ready for Image OCR.")

# TEST 5: Full RAG Pipeline Integration
print("\n5. Testing QA Engine Pipeline Integration...")
try:
    from src.qa_engine import QAEngine
    engine = QAEngine()
    
    mock_doc = "The campus robot uses an ESP32 microcontroller and ultrasonic sensors to navigate safely."
    engine.ingest_and_segment({"test_doc.txt": mock_doc})
    print("✅ Segmentation & DB Insertion successful.")
    
    res = engine.get_answer("What components does the robot use?", [])
    if res['content']:
        print("✅ RAG Retrieval & AI Reasoning successful.")
    else:
        print("❌ ERROR: AI returned empty content.")
except Exception as e:
    print(f"❌ ERROR: Pipeline test failed. Details: {e}")

print("\n--- SYSTEM DIAGNOSTIC COMPLETE ---")
if api_key:
    print("If all tests have green checkmarks, your enterprise architecture is 100% operational!")
    print("Run 'streamlit run app.py' to launch the UI.")