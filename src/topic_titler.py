from typing import List, Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.logging_utils import get_logger

# Resource initialization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

try:
    nltk.data.find("chunkers/maxent_ne_chunker")
except LookupError:
    nltk.download("maxent_ne_chunker", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet

# Custom academic stop-phrase list
STOP_PHRASES = {
    "this segment discusses", "this section discusses", "furthermore we see", 
    "as shown in", "the results suggest", "it is clear that", "we can observe",
    "the following table", "in this study", "related work", "document overview",
    "switching topics", "let us examine"
}

# Define NP Chunk grammar
NP_GRAMMAR = r"""
  NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}   # chunk determiner/possessive, adjectives and nouns
      {<NNP>+}                    # chunk sequences of proper nouns
"""
NP_PARSER = nltk.RegexpParser(NP_GRAMMAR)

# Get logger - this ensures logging is configured
logger = get_logger(__name__)


class TopicTitler:
    def __init__(self):
        # Local processing only - no client needed
        self.stop_words = set(stopwords.words("english"))

    def generate_title(self, texts: List[str], doc_full_text: str = None) -> str:
        """Precision refined local title generation with WordNet boosting and tech heuristics."""
        if not texts or not any(text.strip() for text in texts):
            return "General"

        combined_text = " ".join(texts)
        
        # 0. Filter out stop phrases
        cleaned_text_for_phrases = combined_text.lower()
        for phrase in STOP_PHRASES:
            cleaned_text_for_phrases = cleaned_text_for_phrases.replace(phrase, " ")

        # Clean text preserving case and special symbols for tech terms
        clean_text = re.sub(r"[^a-zA-Z0-9\s\-\.]", " ", combined_text)
        tokens = word_tokenize(clean_text)

        # 1. NP Chunking with enhanced extraction
        tokens = word_tokenize(clean_text)
        if not tokens:
            return "General"

        tagged = nltk.pos_tag(tokens)
        try:
            tree = NP_PARSER.parse(tagged)
            
            noun_phrases = []
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                phrase_tokens = [word for word, tag in subtree.leaves()]
                phrase = " ".join(phrase_tokens)
                if len(phrase.split()) >= 1 and phrase.lower() not in self.stop_words:
                    noun_phrases.append(phrase)
        except Exception as e:
            logger.warning(f"NP Chunking failed: {e}")
            noun_phrases = []

        # 2. Semantic Weighting (TF-IDF + WordNet Boost)
        segment_keywords = [w.lower() for w in tokens if w.lower() not in self.stop_words and len(w) > 2]
        word_scores = Counter()
        
        # Base TF-IDF/Position score
        if doc_full_text:
            doc_tokens = word_tokenize(doc_full_text.lower())
            doc_freq = Counter(doc_tokens)
            seg_freq = Counter(segment_keywords)
            for word, count in seg_freq.items():
                tf = count
                idf = 1.0 / (1.0 + doc_freq[word])
                try:
                    first_idx = segment_keywords.index(word)
                    pos_weight = 1.0 / (1.0 + (first_idx / len(segment_keywords)))
                except ValueError: pos_weight = 1.0
                word_scores[word] = tf * idf * pos_weight
        else:
            for i, word in enumerate(segment_keywords):
                word_scores[word] += 1.0 / (1.0 + (i / len(segment_keywords)))

        # Semantic Boost using WordNet
        try:
            unique_keywords = list(word_scores.keys())
            for i in range(len(unique_keywords)):
                w1 = unique_keywords[i]
                syns1 = wordnet.synsets(w1)
                if not syns1: continue
                
                for j in range(i + 1, len(unique_keywords)):
                    w2 = unique_keywords[j]
                    syns2 = wordnet.synsets(w2)
                    if not syns2: continue
                    
                    # If they share a synset or are very similar, boost both
                    # Using path similarity as a proxy for relatedness
                    sim = syns1[0].path_similarity(syns2[0])
                    if sim and sim > 0.2:
                        word_scores[w1] += 0.2 * sim
                        word_scores[w2] += 0.2 * sim
        except Exception as e:
            logger.warning(f"WordNet boost failed: {e}")

        for word, tag in tagged:
            if (word.isupper() and len(word) > 1) or (tag in ['NNP', 'NNPS']):
                word_scores[word.lower()] += 1.0

        weighted_phrases = []
        for phrase in noun_phrases:
            phrase_words = phrase.lower().split()
            if not phrase_words: continue
            if any(pw in self.stop_words for pw in phrase_words): continue
            
            # Base score from word_scores
            phrase_score = sum(word_scores.get(w, 0) for w in phrase_words)
            
            # Heuristic: Prefer longer but specific phrases
            if len(phrase_words) > 1: phrase_score += 0.5
            
            # Tech weight
            for word in phrase.split():
                if word.isupper() and len(word) > 1: phrase_score += 2.0 # Increased boost
                elif word[0].isupper(): phrase_score += 0.8 # Increased boost

            weighted_phrases.append((phrase, phrase_score))
        
        weighted_phrases.sort(key=lambda x: x[1], reverse=True)
        
        if weighted_phrases:
            best_phrase, best_score = weighted_phrases[0]
            
            # Intelligent capitalization restorer & Acronym Preservation
            result_words = []
            orig_tokens = word_tokenize(combined_text)
            orig_lower = [t.lower() for t in orig_tokens]
            
            for w in best_phrase.split():
                try:
                    # Look for exact match first to capture technical casing
                    matches = [orig_tokens[i] for i, val in enumerate(orig_lower) if val == w.lower()]
                    
                    # Heuristic: Pick the most "technical" looking version (most uppercase)
                    if matches:
                        orig_case = max(matches, key=lambda x: sum(1 for c in x if c.isupper()))
                    else:
                        orig_case = w

                    # If it's an acronym (all caps) or proper start, keep original case
                    # Otherwise, capitalize only if it's not a stop word
                    if orig_case.isupper() and len(orig_case) > 1:
                        result_words.append(orig_case)
                    elif orig_case[0].isupper() and w.lower() not in self.stop_words:
                        result_words.append(orig_case)
                    else:
                        result_words.append(w.capitalize())
                except Exception:
                    result_words.append(w.capitalize())
            
            # Auto-scaling title length: 2-5 words based on phrase coherence
            # But ensure we don't truncate a short valid technical phrase
            max_words = min(5, len(result_words))
            if len(result_words) <= 5:
                final_title = " ".join(result_words)
            else:
                final_title = " ".join(result_words[:max_words])
            
            # Remove trailing punctuation or generic filler
            final_title = re.sub(r"[\.\s]+$", "", final_title)
            return final_title

        top_keywords = [w for w, _ in word_scores.most_common(2)]
        return " ".join(top_keywords).title() if top_keywords else "General"
