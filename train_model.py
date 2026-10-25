import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from preprocess import clean_text

def train_models():
    print("Loading dataset...")
    data_path = os.path.join('data', 'feedback.csv')
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    if df.empty:
        print("Dataset is empty. Add data to feedback.csv.")
        return
        
    print("Vectorizing text using TF-IDF...")
    # Use bigrams to capture "very fast", "not working", etc.
    vectorizer = TfidfVectorizer(
        max_features=1000,
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    # Train Sentiment Model
    print("\n--- Training Sentiment Classifier ---")
    y_sentiment = df['sentiment']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )
    
    sentiment_model = LogisticRegression(C=1000, max_iter=200, class_weight='balanced')
    sentiment_model.fit(X_train_s, y_train_s)
    
    sentiment_preds = sentiment_model.predict(X_test_s)
    print("Sentiment Model Evaluation (Test Data):")
    print(f"Accuracy: {accuracy_score(y_test_s, sentiment_preds):.2f}")
    print(classification_report(y_test_s, sentiment_preds))
    
    # Train Category Model
    print("\n--- Training Category Classifier ---")
    y_category = df['category']
    
    # Check if we can stratify on category (some classes might have only 1 sample in a very small dataset, but ours is balanced enough)
    try:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
    except ValueError:
        # Fallback if stratify fails due to single-instance classes
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_category, test_size=0.2, random_state=42
        )
        
    category_model = LogisticRegression(C=1000, max_iter=200, class_weight='balanced')
    category_model.fit(X_train_c, y_train_c)
    
    category_preds = category_model.predict(X_test_c)
    print("Category Model Evaluation (Test Data):")
    print(f"Accuracy: {accuracy_score(y_test_c, category_preds):.2f}")
    print(classification_report(y_test_c, category_preds))
    
    # Retrain on full data for production usage in the mini-project
    # so we use all 100% of our small dataset for the actual web app
    print("\nRetraining models on full dataset for final application...")
    sentiment_model_final = LogisticRegression(C=1000, max_iter=200, class_weight='balanced')
    sentiment_model_final.fit(X, y_sentiment)
    
    category_model_final = LogisticRegression(C=1000, max_iter=200, class_weight='balanced')
    category_model_final.fit(X, y_category)
    
    print("Saving models to disk...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, os.path.join('models', 'vectorizer.pkl'))
    joblib.dump(sentiment_model_final, os.path.join('models', 'sentiment_model.pkl'))
    joblib.dump(category_model_final, os.path.join('models', 'category_model.pkl'))
    
    print("Training complete! Models saved in 'models/' directory.")

if __name__ == "__main__":
    train_models()
