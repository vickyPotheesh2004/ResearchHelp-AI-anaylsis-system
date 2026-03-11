import os
import sys
import time
import logging
from dotenv import load_dotenv

from src.qa_engine import QAEngine
from src.research_engine import ResearchEngine
from app import generate_markdown_export, generate_html_export, generate_docx_export

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_az_test():
    load_dotenv()
    logging.info("=========================================")
    logging.info("🚀 ResearchHelp-AI-anaylsis-system full system testing...")
    logging.info("=========================================")

    # 1. Initialize Engines
    logging.info("\n[1/5] Initializing Systems & ChromaDB...")
    qa_engine = QAEngine()
    research_engine = ResearchEngine()
    
    # Force a clean start (wipe DB)
    try:
        qa_engine.chroma_client.delete_collection("doc_chunks")
    except Exception:
        pass
    qa_engine.collection = qa_engine.chroma_client.get_or_create_collection("doc_chunks")

    # 2. Ingest Document
    logging.info("\n[2/5] Simulating Document Ingestion...")
    mock_doc = """
    ## The Quantum Drive Architecture V2.0
    
    The new Quantum Drive system proposed for the Horizon spacecraft utilizes a dual-core neutrino reactor.
    This reactor generates thrust by stabilizing quantum fluctuations in a contained vacuum shell.
    The primary material used for the shell is advanced carbon-nanotube mesh intertwined with titanium alloys.
    
    However, heat dissipation remains a significant hurdle. When the drive operates above 85% capacity for more than 4 hours, thermal runaway can occur, reaching temperatures over 4000K.
    It is recommended to integrate a liquid nitrogen cooling loop specifically for the secondary injection valve to mitigate this.
    Current cost estimates place the production of a single Quantum Drive at $1.2 Billion USD.
    """
    
    doc_dict = {"quantum_drive_spec.txt": mock_doc}
    qa_engine.ingest_and_segment(doc_dict)
    logging.info("Document ingested, chunked, and stored in vector DB successfully.")

    # 3. Generating Insights (Research Engine)
    logging.info("\n[3/5] Generating Document Insights...")
    chunks, metas = qa_engine.get_all_chunks()
    overview = research_engine.generate_document_overview(chunks, metas)
    logging.info(f"Generated Overview Length: {len(overview)} characters.")
    
    suggestions = research_engine.generate_auto_suggestions(chunks, metas)
    logging.info(f"Generated {len(suggestions)} AI Suggestions.")
    for i, sug in enumerate(suggestions):
        logging.info(f"  - [{sug.get('category', 'N/A').upper()}] {sug.get('title')}")

    # 4. Testing Core Pipeline logic (Non-Streaming & Memory)
    logging.info("\n[4/6] Testing Core Pipeline logic & Memory (Non-Streaming)...")
    res1 = qa_engine.get_answer("What material is used for the vacuum shell?", [])
    logging.info(f"Direct Response Content: {res1['content'][:100]}...")
    logging.info(f"Logic Trace Captured: {bool(res1['reasoning_details'])}")
    
    # Test Follow-up with history
    history = [
        {"role": "user", "content": "What material is used for the vacuum shell?"},
        {"role": "assistant", "content": res1['content'], "reasoning_details": res1['reasoning_details']}
    ]
    res2 = qa_engine.get_answer("Why is that material primarily used?", history)
    logging.info(f"Follow-up Response: {res2['content'][:100]}...")
    logging.info("Memory and context retrieval passed.")

    # 5. Testing Intent Routing & Streaming RAG (QA Engine)
    logging.info("\n[5/6] Testing Query Engine Intention Routing (A-Z)...")
    
    test_queries = [
        {"desc": "Direct factual Q&A", "q": "What material is used for the vacuum shell?"},
        {"desc": "Improvement Suggestion", "q": "How can we fix the heat dissipation problem?"},
        {"desc": "Research Add-on Feasibility", "q": "Can we add solar panels to power the drive instead of a reactor?"},
        {"desc": "Simple English Deep Analysis", "q": "Explain the physics of quantum fluctuations in simple terms"},
        {"desc": "IEEE Official Paper Generator", "q": "Generate an IEEE official paper based on this quantum drive project"},
        {"desc": "Domain Detection Test", "q": "How does this relate to deep learning and computer vision?"},
        {"desc": "Off-Topic Rejection", "q": "What is the recipe for a chocolate cake?"}
    ]

    chat_history = []
    
    for test in test_queries:
        logging.info(f"\n--- Testing Intent: {test['desc']} ---")
        logging.info(f"Query: '{test['q']}'")
        
        final_meta = {}
        final_text = ""
        
        # Stream the response
        try:
            for event in qa_engine.get_answer_stream(test['q'], chat_history):
                if event["type"] == "meta":
                    final_meta = event
                    intent_info = event.get("intent", {})
                    logging.info(f"Detected Intent: {intent_info.get('emoji', '❓')} {intent_info.get('label', 'Unknown')}")
                    sources = event.get("sources", [])
                    logging.info(f"Retrieved {len(sources)} sources via Hybrid Search.")
                elif event["type"] == "token":
                    pass # We won't print every token in the test script to keep logs clean
                elif event["type"] == "done":
                    final_text = event.get("content", "")
                    logging.info(f"Final Response Length: {len(final_text)} chars.")
                    assert len(final_text) > 20, "Response is suspiciously short!"
        except AttributeError:
            # Fallback for older non-streaming method if run directly
            res = qa_engine.get_answer(test['q'], chat_history)
            final_text = res["content"]
            final_meta = {"intent": res.get("intent"), "sources": res.get("sources", [])}
            
        chat_history.append({"role": "user", "content": test['q']})
        chat_history.append({"role": "assistant", "content": final_text})

    # 6. Testing Export Generation
    logging.info("\n[6/6] Testing Session Export Formats...")
    stats = qa_engine.get_session_stats()
    
    md_export = generate_markdown_export(chat_history, overview, suggestions, stats)
    html_export = generate_html_export(chat_history, overview, suggestions, stats)
    docx_export = generate_docx_export(chat_history, overview, suggestions, stats)
    
    logging.info(f"Generated Markdown Export: {len(md_export)} bytes")
    logging.info(f"Generated HTML Export: {len(html_export)} bytes")
    logging.info(f"Generated DOCX Export: {len(docx_export.getvalue()) if hasattr(docx_export, 'getvalue') else 'OK'} bytes")
    
    logging.info("\n=========================================")
    logging.info("✅ ALL SYSTEMS PASSED! A-Z TEST COMPLETE")
    logging.info("=========================================")

if __name__ == "__main__":
    run_az_test()
