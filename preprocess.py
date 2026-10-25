import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

# Do not remove important sentiment and negation words
important_words_to_keep = {
    'not', 'no', 'nor', 'very', 'well', 'too', 'good', 'bad', 'fast', 'slow', 
    'excellent', 'poor', 'great', 'terrible', 'worst', 'best', 'working', 
    'broken', 'dirty', 'clean', 'helpful', 'rude', 'late', 'fresh', 'tasty', 'delicious'
}
stop_words = stop_words - important_words_to_keep

stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans the input text by applying standard NLP preprocessing steps.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    
    # Stemming is generally good, but sometimes it ruins 'not working' -> 'not work' 
    # Let's keep it simple with PorterStemmer as requested, but ensure keywords remain intact for insight.
    clean_tokens = [stemmer.stem(word) if word not in important_words_to_keep else word for word in tokens if word not in stop_words]
    
    return " ".join(clean_tokens)

def extract_keywords(text):
    """
    Extracts important words directly from text for display and insight generation.
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    
    # Remove stopwords but keep original word forms for readability
    keywords = [word for word in tokens if word not in stop_words and len(word) > 2]
    return list(dict.fromkeys(keywords))
