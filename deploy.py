from flask import Flask, request, jsonify
import pickle
import numpy as np
import re
from collections import Counter

with open('naive_bayes_model.pkl', 'rb') as f:
    log_class_priors, log_word_probs = pickle.load(f)

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('doc_freq.pkl', 'rb') as f:
    doc_freq = pickle.load(f)

# Preprocessing function (same as training)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    return words

def predict_sentiment(text):
    words = preprocess(text)
    word_counts = Counter(words)  

    class_scores = {0: log_class_priors[0], 1: log_class_priors[1]} 

    for class_label in [0, 1]:
        score = class_scores[class_label] 

        for word, count in word_counts.items():  
            if word in vocab: 
                word_idx = vocab[word]  

                score += count * log_word_probs[class_label][word_idx]
            else:
                score += count * (-10)  

        class_scores[class_label] = score  

    predicted_class = max(class_scores, key=class_scores.get)
    return "positive" if predicted_class == 1 else "negative"




app = Flask(__name__)

@app.route('/')
def home():
    return "Custom Naive Bayes Sentiment Classifier is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'Review text is required'}), 400

    sentiment = predict_sentiment(review)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
