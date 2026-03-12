import re
import logging
import nltk

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logger.warning("NLTK punkt tokenizer not found. Downloading...")
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.warning("NLTK punkt_tab not found. Downloading...")
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass  # Some NLTK versions don't have punkt_tab

FILLERS = [
    r"\byou know\b",
    r"\bokay\b",
    r"\balright\b",
    r"\buh\b",
    r"\bum\b",
    r"\byeah\b",
    r"\bright\b",
    r"\bjust\b",
    r"\bi mean\b",
]

ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "inc.",
    "ltd.",
    "dept.",
    "approx.",
    "st.",
    "ave.",
    "blvd.",
    "fig.",
    "eq.",
    "ref.",
    "vol.",
    "no.",
    "pp.",
    "ed.",
    "rev.",
    "e.g.",
    "i.e.",
}


def smart_sentence_split(text: str) -> list:
    """Split text into sentences intelligently, handling abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()

    raw_splits = re.split(r"(?<=[.!?])\s+", text)

    sentences = []
    buffer = ""

    for fragment in raw_splits:
        buffer = (buffer + " " + fragment).strip() if buffer else fragment

        last_word_match = re.search(r"(\S+)\s*$", buffer)
        if last_word_match:
            last_word = last_word_match.group(1).lower()
            if last_word in ABBREVIATIONS:
                continue

        if len(buffer) > 20:
            sentences.append(buffer)
            buffer = ""

    if buffer and len(buffer.strip()) > 10:
        sentences.append(buffer.strip())

    return sentences


def clean_text(text: str) -> str:
    """Remove filler words and clean up text."""
    t = text
    for f in FILLERS:
        t = re.sub(f, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def preserve_structure(text: str) -> str:
    """Preserve document structure while normalizing whitespace."""
    lines = text.split("\n")
    processed = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(#{1,6}\s|[\d]+\.|[-*•]\s)", stripped):
            processed.append(f"\n{stripped}\n")
        else:
            processed.append(stripped)
    return " ".join(processed)
