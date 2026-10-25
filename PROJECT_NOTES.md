# Project Notes: CampusVoice (NLP Viva Guide)

## Problem Statement
Colleges and educational institutions receive large amounts of unstructured feedback from students. Manually reading, categorizing, and prioritizing this feedback is time-consuming and inefficient.

## Objective
To build an automated Natural Language Processing (NLP) pipeline that can ingest student feedback and automatically extract sentiment (Positive/Negative/Neutral), categorize the domain (Academics, Infrastructure, etc.), and determine issue priority.

## NLP Techniques Used
1. **Text Cleaning**: Removing special characters, numbers, and converting to lowercase.
2. **Tokenization**: Splitting the sentence into individual words (tokens).
3. **Stop Word Removal**: Removing common, low-meaning words like "is", "the", "and" using NLTK to focus on words that carry actual semantic meaning.
4. **Stemming (Porter Stemmer)**: Reducing words to their root form (e.g., "running" to "run") to standardize variations.
5. **Vectorization (TF-IDF)**: Term Frequency-Inverse Document Frequency. This converts text into numbers (vectors) so the ML algorithm can understand it. It's better than simple word counting (Bag of Words) because it penalizes highly frequent words that appear in all documents and gives higher weight to unique, meaningful words.

## ML Algorithm: Logistic Regression
- **Why Logistic Regression?**: It is a simple, fast, and highly interpretable classification algorithm. It works very well for linearly separable data, which text features (TF-IDF vectors) often are. It's perfectly suited for a baseline NLP classifier.
- We use two separate Logistic Regression models: one for Sentiment, one for Category.

## Dataset
- **Training Data**: A custom-made CSV (`feedback.csv`) containing sample college feedback mapped to sentiment and category labels.
- **Testing Data**: In this prototype, we use the model to predict on user input (unseen data) entered through the web interface.

## Features
- Complete NLP processing pipeline.
- Dual-model ML architecture.
- Explainable NLP Insights showing keyword associations.
- Actionable priority rules.
- Analytics dashboard.

## Limitations
- **Small Dataset**: The model is trained on a very small sample dataset (around 50 rows) for demonstration. It may misclassify complex or completely new phrases.
- **Sarcasm Detection**: The current TF-IDF and Logistic Regression pipeline cannot reliably detect sarcasm or complex context.
- **Rule-based Priority**: Priority is currently calculated using hardcoded keywords rather than a learned model.

## Future Scope
- Expand the dataset to thousands of real feedback entries.
- Use advanced word embeddings (Word2Vec or GloVe) instead of TF-IDF for better semantic understanding.
- Implement deep learning (LSTMs or Transformers) for context-aware classification.
- Integrate with a real college database/ticketing system to automatically assign tasks to departments.

## How the Prediction Pipeline Works
User Input -> Text Cleaning -> Stop Words Removed -> Stemming -> TF-IDF Vectorization -> Logistic Regression Model -> Output (Sentiment + Category) -> Rule Engine (Priority) -> Frontend Display.
