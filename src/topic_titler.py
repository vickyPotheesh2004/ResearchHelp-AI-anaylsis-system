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
    "this section discusses", "furthermore we see", "as shown in", 
    "the results suggest", "it is clear that", "we can observe",
    "the following table", "in this study", "related work"
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

        if not tokens:
            return "General"

        # 1. NP Chunking with enhanced extraction
        try:
            tagged = nltk.pos_tag(tokens)
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

        # 3. Final Selection with Phrase Heuristics
        weighted_phrases = []
        for phrase in noun_phrases:
            phrase_words = phrase.lower().split()
            if not phrase_words: continue
            
            # Base score from word_scores
            phrase_score = sum(word_scores.get(w, 0) for w in phrase_words) / len(phrase_words)
            
            # Heuristic: Technical term detection (Preserve capitalization like 'GPU', 'BERT')
            # If a word is mostly uppercase or starts with uppercase in original text, it might be a tech term
            for word in phrase.split():
                if word.isupper() and len(word) > 1:
                    phrase_score += 0.5
                elif word[0].isupper() and word.lower() not in self.stop_words:
                    phrase_score += 0.2

            if len(phrase_words) > 1: phrase_score *= 1.3
            weighted_phrases.append((phrase, phrase_score))
        
        weighted_phrases.sort(key=lambda x: x[1], reverse=True)
        
        if weighted_phrases:
            best_phrase = weighted_phrases[0][0]
            # Intelligent capitalization restorer
            result_words = []
            orig_tokens = word_tokenize(combined_text)
            orig_lower = [t.lower() for t in orig_tokens]
            
            for w in best_phrase.split():
                try:
                    idx = orig_lower.index(w.lower())
                    orig_case = orig_tokens[idx]
                    # If it's an acronym or proper start, keep original case
                    if orig_case.isupper() or (orig_case[0].isupper() and w.lower() not in self.stop_words):
                        result_words.append(orig_case)
                    else:
                        result_words.append(w.capitalize())
                except ValueError:
                    result_words.append(w.capitalize())
            
            final_title = " ".join(result_words[:3])
            return final_title

        top_keywords = [w for w, _ in word_scores.most_common(2)]
        return " ".join(top_keywords).title() if top_keywords else "General"
