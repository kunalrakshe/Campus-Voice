# CampusVoice: Intelligent Student Feedback Analyzer

A real-world Natural Language Processing (NLP) application for analyzing student feedback in colleges. It automatically determines the sentiment, categorizes the department involved, extracts keywords, and assigns a priority using Machine Learning (TF-IDF and Logistic Regression).

## Features
- **NLP Pipeline**: Includes text cleaning, tokenization, stop-word removal, and TF-IDF vectorization.
- **Machine Learning**: Uses Logistic Regression classifiers trained from scratch (no external APIs).
- **Rule-based Priority**: Calculates issue urgency (High, Medium, Low) based on key terms.
- **Explainable Insights**: Provides simple text-based reasoning for its classification (NLP Insight).
- **Analytics Dashboard**: Visualizes feedback trends.

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the ML Models**:
   The `data/feedback.csv` file contains the training dataset. Run the training script to generate the `.pkl` model files:
   ```bash
   python train_model.py
   ```

3. **Run the Application**:
   Start the Flask web server:
   ```bash
   python app.py
   ```

4. **Access the Interface**:
   Open a web browser and go to: `http://127.0.0.1:5000`

## Important Notes
- This is a student mini-project prototype.
- The dataset (`feedback.csv`) is small for demonstration purposes. In a production environment, this would require thousands of rows for high accuracy.
