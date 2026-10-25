from flask import Flask, render_template, request, redirect, url_for
from model import analyzer
import json
import os
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = os.path.join('data', 'history.json')

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('feedback_text', '')
    if not text.strip():
        return redirect(url_for('index'))
        
    result = analyzer.analyze(text)
    
    # Save to history
    history = load_history()
    record = {
        'id': len(history) + 1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),gi
        'text': text,
        'sentiment': result['sentiment'],
        'sentiment_confidence': result['sentiment_confidence'],
        'category': result['category'],
        'priority': result['priority']
    }
    history.insert(0, record) # Insert at beginning
    save_history(history)
    
    return render_template('result.html', text=text, result=result)

@app.route('/history')
def history():
    records = load_history()
    return render_template('history.html', history=records)

@app.route('/dashboard')
def dashboard():
    records = load_history()
    
    total = len(records)
    positive = sum(1 for r in records if r['sentiment'] == 'Positive')
    negative = sum(1 for r in records if r['sentiment'] == 'Negative')
    neutral = sum(1 for r in records if r['sentiment'] == 'Neutral')
    high_priority = sum(1 for r in records if r['priority'] == 'High')
    
    # Calculate most reported category
    categories = {}
    for r in records:
        cat = r['category']
        categories[cat] = categories.get(cat, 0) + 1
        
    most_reported = "None"
    if categories:
        most_reported = max(categories, key=categories.get)
        
    stats = {
        'total': total,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'high_priority': high_priority,
        'most_reported': most_reported,
        'categories': categories
    }
    
    return render_template('dashboard.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
