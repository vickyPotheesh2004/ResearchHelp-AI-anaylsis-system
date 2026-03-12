from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import threading
from src.topic_titler import TopicTitler
from src.text_preprocessor import smart_sentence_split
from src.config import (
    TOPIC_SIMILARITY_THRESHOLD,
    TOPIC_SEGMENT_OVERLAP,
    EMBEDDING_MODEL,
)
from src.logging_utils import get_logger

# Get logger
logger = get_logger(__name__)

# Thread-safe singleton model instance
_SENTENCE_MODEL = None
_model_lock = threading.Lock()


def _get_sentence_model():
    """Get or create the thread-safe singleton SentenceTransformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        with _model_lock:
            # Double-check locking pattern
            if _SENTENCE_MODEL is None:
                _SENTENCE_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _SENTENCE_MODEL


class TopicSegmenter:
    def __init__(self):
        self.model = _get_sentence_model()
        self.threshold = TOPIC_SIMILARITY_THRESHOLD
        self.overlap = TOPIC_SEGMENT_OVERLAP
        self.titler = TopicTitler()

    def segment(self, text: str) -> List[Dict]:
        sentences = smart_sentence_split(text)
        sentences = [s for s in sentences if len(s.strip()) > 15]
        if not sentences:
            return []

        embeddings = self.model.encode(sentences, show_progress_bar=False)
        topics = []
        current_sentences = [sentences[0]]
        topic_id = 0

        for i in range(1, len(embeddings)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]

            if sim < self.threshold and len(current_sentences) >= 2:
                title = self.titler.generate_title(current_sentences)
                topics.append(
                    {
                        "topic_id": topic_id,
                        "title": title,
                        "content": " ".join(current_sentences),
                    }
                )
                topic_id += 1

                overlap_sentences = (
                    current_sentences[-self.overlap :]
                    if len(current_sentences) >= self.overlap
                    else []
                )
                current_sentences = overlap_sentences + [sentences[i]]
            else:
                current_sentences.append(sentences[i])

        if current_sentences:
            title = self.titler.generate_title(current_sentences)
            topics.append(
                {
                    "topic_id": topic_id,
                    "title": title,
                    "content": " ".join(current_sentences),
                }
            )

        return topics
