import joblib
import os
import re
import numpy as np
from preprocess import clean_text, extract_keywords

class FeedbackAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.sentiment_model = None
        self.category_model = None
        self.load_models()
        
        self.high_priority_terms = ['not working', 'broken', 'unsafe', 'dangerous', 'urgent']
        self.medium_priority_terms = ['slow', 'poor', 'problem', 'needs improvement']
        
        self.strong_positive = ['very good', 'excellent', 'very fast', 'working perfectly', 'great', 'helpful', 'well maintained', 'clear explanation', 'good service', 'quite good']
        self.strong_negative = ['very slow', 'not working', 'broken', 'terrible', 'worst', 'very bad', 'poor service', 'not helpful', 'dirty', 'loud noise', 'too high']
        
        self.contrast_words = ['but', 'however', 'although', 'though']
        self.basic_pos_words = ['good', 'great', 'excellent', 'fast', 'helpful', 'polite', 'tasty', 'delicious', 'clean', 'working', 'perfect', 'awesome', 'nice', 'clear']
        self.basic_neg_words = ['bad', 'terrible', 'poor', 'slow', 'unreliable', 'noisy', 'broken', 'dirty', 'late', 'rude', 'awful', 'stale', 'worst', 'not']

    def load_models(self):
        try:
            self.vectorizer = joblib.load(os.path.join('models', 'vectorizer.pkl'))
            self.sentiment_model = joblib.load(os.path.join('models', 'sentiment_model.pkl'))
            self.category_model = joblib.load(os.path.join('models', 'category_model.pkl'))
        except Exception as e:
            print(f"Warning: Models not loaded. Please run train_model.py first. Error: {e}")

    def calculate_priority(self, text):
        text_lower = text.lower()
        
        for term in self.high_priority_terms:
            if term in text_lower:
                return 'High'
                
        for term in self.medium_priority_terms:
            if term in text_lower:
                return 'Medium'
                
        return 'Low'

    def hybrid_sentiment_check(self, text, ml_sentiment):
        text_lower = text.lower()
        
        if "not bad" in text_lower:
            return "Positive", "Hybrid Rule: 'not bad' detected."
            
        for phrase in self.strong_negative:
            if phrase in text_lower:
                return "Negative", f"Hybrid Rule: Strong negative phrase '{phrase}' detected."
                
        for phrase in self.strong_positive:
            if phrase in text_lower:
                return "Positive", f"Hybrid Rule: Strong positive phrase '{phrase}' detected."
                
        return ml_sentiment, "ML Model Prediction"

    def generate_insight(self, text, sentiment, category, sentiment_conf):
        text_lower = text.lower()
        found_pos = [p for p in self.strong_positive if p in text_lower]
        found_neg = [p for p in self.strong_negative if p in text_lower]
        
        insight_parts = []
        
        if sentiment_conf < 60.0:
            insight_parts.append(f"Low confidence ({sentiment_conf}%): the model is uncertain about this feedback or it contains mixed sentiments.")
        
        if found_pos or found_neg:
            terms = found_pos + found_neg
            term_str = ", ".join([f"'{t}'" for t in terms[:3]])
            if sentiment == 'Positive':
                insight_parts.append(f"Positive phrase {term_str} was detected.")
            elif sentiment == 'Negative':
                insight_parts.append(f"Negative phrase {term_str} was detected.")
        else:
            insight_parts.append(f"The ML model classified the sentiment as {sentiment} based on the overall word usage.")
            
        insight_parts.append(f"The terms indicate the {category} category.")
        
        return " ".join(insight_parts)
        
    def get_suggested_action(self, category, priority, sentiment):
        if sentiment == 'Positive':
            return "No immediate action required. Maintain current standards."
            
        if priority == 'High':
            return f"URGENT: Forward to the head of {category} department for immediate inspection and resolution."
        elif priority == 'Medium':
            return f"Log a ticket with the {category} department and schedule for review within 3-5 days."
        else:
            return f"Monitor feedback related to {category} for any emerging trends."

    def analyze(self, text):
        if not self.vectorizer:
            self.load_models()
            
        cleaned = clean_text(text)
        keywords = extract_keywords(text)
        
        if not cleaned.strip():
             return {
                 'sentiment': 'Neutral',
                 'sentiment_confidence': 0.0,
                 'category': 'Other',
                 'category_confidence': 0.0,
                 'priority': 'Low',
                 'cleaned_text': cleaned,
                 'keywords': [],
                 'insight': 'Text was empty or only contained stop words.',
                 'action': 'None'
             }
             
        X = self.vectorizer.transform([cleaned])
        
        ml_sentiment = self.sentiment_model.predict(X)[0]
        sentiment_probs = self.sentiment_model.predict_proba(X)[0]
        sentiment_conf = round(np.max(sentiment_probs) * 100, 1)
        
        category = self.category_model.predict(X)[0]
        category_probs = self.category_model.predict_proba(X)[0]
        category_conf = round(np.max(category_probs) * 100, 1)
        
        # Hybrid Sentiment Layer
        final_sentiment, hybrid_reason = self.hybrid_sentiment_check(text, ml_sentiment)
        
        # Adjust confidence representation if hybrid layer kicks in
        if final_sentiment != ml_sentiment:
            sentiment_conf = 100.0
        elif hybrid_reason != "ML Model Prediction":
            # ML was right, but hybrid rule also confirms it
            sentiment_conf = max(sentiment_conf, 90.0)
            
        # Priority is independent of sentiment/category
        priority = self.calculate_priority(text)
        
        insight = self.generate_insight(text, final_sentiment, category, sentiment_conf)
        
        action = self.get_suggested_action(category, priority, final_sentiment)
        
        # Check if mixed feedback (low confidence)
        if sentiment_conf < 45.0 and final_sentiment == ml_sentiment:
            final_sentiment = "Mixed / Uncertain"
            
        # Lightweight mixed-sentiment check for contrasting structures
        text_lower = text.lower()
        # Clean punctuation for word matching
        words = re.findall(r'\b\w+\b', text_lower)
        has_contrast = any(word in words for word in self.contrast_words)
        if has_contrast:
            has_pos = any(word in words for word in self.basic_pos_words)
            has_neg = any(word in words for word in self.basic_neg_words)
            if has_pos and has_neg:
                final_sentiment = "Mixed / Uncertain"
                insight = "Contrast word detected with both positive and negative language. Sentiment is Mixed / Uncertain."
        
        
        return {
            'sentiment': final_sentiment,
            'sentiment_confidence': sentiment_conf,
            'category': category,
            'category_confidence': category_conf,
            'priority': priority,
            'cleaned_text': cleaned,
            'keywords': keywords,
            'insight': insight,
            'action': action
        }

analyzer = FeedbackAnalyzer()
