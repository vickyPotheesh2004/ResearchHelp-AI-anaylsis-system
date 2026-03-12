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
        if len(sentences) < 4:
            # Too short for complex segmentation
            return [{"topic_id": 0, "title": self.titler.generate_title(sentences, text), "content": text}] if sentences else []

        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # 1. Calculate similarity profile (gaps between sentences)
        similarities = []
        for i in range(len(embeddings) - 1):
            # Use a small window for local coherence
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            
        # 2. Calculate Depth Scores (Inspired by TextTiling)
        # Depth score measures how much a "valley" in similarity compares to its neighbors
        depth_scores = [0.0] * len(similarities)
        for i in range(len(similarities)):
            # Scan left for peak
            lpeak = similarities[i]
            for j in range(i - 1, -1, -1):
                if similarities[j] >= lpeak: lpeak = similarities[j]
                else: break
            
            # Scan right for peak
            rpeak = similarities[i]
            for j in range(i + 1, len(similarities)):
                if similarities[j] >= rpeak: rpeak = similarities[j]
                else: break
                
            depth_scores[i] = (lpeak - similarities[i]) + (rpeak - similarities[i])

        # 3. Identify boundaries (Statistical Outlier Detection)
        # Instead of a fixed threshold, we look for "anomalies" in thematic coherence
        if not depth_scores:
            return [{"topic_id": 0, "title": self.titler.generate_title(sentences, text), "content": text}]

        avg_depth = sum(depth_scores) / len(depth_scores)
        variance = sum((d - avg_depth)**2 for d in depth_scores) / len(depth_scores)
        std_dev = variance**0.5
        
        # Perfection Heuristic: Sensitivity scales with document complexity
        # For very 'flat' documents, we need higher sensitivity to catch subtle shifts.
        # We also lower the threshold slightly to be more aggressive in finding boundaries.
        sensitivity_factor = 0.8 + (0.1 if std_dev < 0.1 else 0)
        boundary_threshold = avg_depth + (sensitivity_factor * std_dev)
        
        # Minimum segment length also scales with document size (Auto-scaling)
        # We allow for shorter segments (minimum 1 sentence) if they are highly distinct.
        dynamic_min_len = max(1, min(3, len(sentences) // 20))
        
        potential_boundaries = sorted(
            [(i, depth_scores[i]) for i in range(len(depth_scores)) if depth_scores[i] > boundary_threshold],
            key=lambda x: x[1],
            reverse=True
        )
        
        actual_boundaries = []
        for idx, score in potential_boundaries:
            if all(abs(idx - b) >= dynamic_min_len for b in actual_boundaries):
                actual_boundaries.append(idx)
                    
        actual_boundaries.sort()
        
        # 4. Construct segments with Dynamic Overlap
        topics = []
        start_idx = 0
        topic_id = 0
        
        # Iterate through boundaries to create segments
        boundary_indices = actual_boundaries + [len(sentences) - 1]
        for i, b_idx in enumerate(boundary_indices):
            end_idx = b_idx + 1
            segment_sentences = sentences[start_idx : end_idx]
            
            if segment_sentences:
                # Use current segment sentences + full document context for TF-IDF
                title = self.titler.generate_title(segment_sentences, text)
                topics.append({
                    "topic_id": topic_id,
                    "title": title,
                    "content": " ".join(segment_sentences)
                })
                topic_id += 1
                
                # Dynamic start for next segment
                start_idx = max(0, end_idx - self.overlap)
        
        # 5. Final Cohesion Merge (Conservative)
        if len(topics) > 1:
            # Perfection Heuristic: We only merge if they are truly identical or tiny (under 10 words)
            refined_topics = []
            for t in topics:
                content_words = t['content'].split()
                if not refined_topics or len(content_words) >= 10:
                    refined_topics.append(t)
                else:
                    # Merge content into previous
                    refined_topics[-1]['content'] += " " + t['content']
                    # Re-generate title for the merged content
                    merged_sents = smart_sentence_split(refined_topics[-1]['content'])
                    refined_topics[-1]['title'] = self.titler.generate_title(merged_sents)
            
            for i, t in enumerate(refined_topics): t['topic_id'] = i
            return refined_topics

        return topics
