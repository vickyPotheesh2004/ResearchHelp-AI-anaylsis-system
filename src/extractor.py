import io
import re
import fitz
import docx
import pandas as pd
import pytesseract
from PIL import Image
from typing import Tuple
from src.config import TESSERACT_PATH, MAX_FILE_SIZE, MAX_TEXT_LENGTH
from src.logging_utils import get_logger

# Get logger
logger = get_logger(__name__)

# Configure Tesseract OCR path from centralized config
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_image_text(image_bytes: bytes) -> str:
    """Extract text from image using OCR."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"[OCR Error: {str(e)}]"


def validate_file_size(file_bytes: bytes) -> Tuple[bool, str]:
    """Validate file size against configured limits."""
    if len(file_bytes) > MAX_FILE_SIZE:
        return (
            False,
            f"Error: File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
        )
    return True, ""


def validate_text_length(text: str) -> str:
    """Validate and truncate text if exceeds maximum length."""
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH] + "\n[Content truncated due to size]"
    return text


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename


def process_file(uploaded_file) -> Tuple[str, str]:
    """Process uploaded file and extract text content.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (sanitized_filename, extracted_text_or_error_message)
    """
    filename = sanitize_filename(uploaded_file.name)
    ext = filename.split(".")[-1].lower()
    text_content = ""
    file_bytes = uploaded_file.read()

    # Validate file size
    is_valid, error_msg = validate_file_size(file_bytes)
    if not is_valid:
        return filename, error_msg

    try:
        if ext in ["png", "jpg", "jpeg"]:
            text_content = extract_image_text(file_bytes)

        elif ext == "txt":
            text_content = file_bytes.decode("utf-8")
            text_content = validate_text_length(text_content)

        elif ext == "pdf":
            # Use context manager for proper file handle management
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text_content += page.get_text() + " "
            text_content = validate_text_length(text_content)

        elif ext == "docx":
            # Use context manager for docx
            with docx.Document(io.BytesIO(file_bytes)) as doc:
                text_content = " ".join([p.text for p in doc.paragraphs])
            text_content = validate_text_length(text_content)

        elif ext in ["csv", "xlsx"]:
            if ext == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
            text_content = df.to_string(index=False)
            text_content = validate_text_length(text_content)

        else:
            return filename, f"Unsupported format: {ext}"

    except UnicodeDecodeError:
        return (
            filename,
            "Error: Unable to decode file. Please use UTF-8 encoded text files.",
        )
    except Exception as e:
        return filename, f"Error: {str(e)}"

    return filename, text_content
