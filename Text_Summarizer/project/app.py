from flask import Flask, render_template, request
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from collections import Counter

app = Flask(__name__)

# Preprocessing
def preprocess(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                 "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
                 "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                 "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
                 "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                 "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
                 "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
                 "between", "into", "through", "during", "before", "after", "above", "below", "to",
                 "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
                 "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
                 "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                 "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
                 "just", "don", "should", "now"]
    preprocessed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        preprocessed_sentence = " ".join(filtered_words)
        preprocessed_sentences.append(preprocessed_sentence)

    return preprocessed_sentences

def extract_features(sentences):
    features = []
    for sentence in sentences:
        sentence_length = len(sentence.split())
        num_words = len(re.findall(r'\w+', sentence))

        if num_words == 0:
            avg_word_length = 0
        else:
            avg_word_length = sum(len(word) for word in sentence.split()) / num_words

        num_characters = len(sentence)
        num_digits = len(re.findall(r'\d', sentence))
        num_special_chars = len(re.findall(r'[^\w\s]', sentence))
        contains_keyword = int('keyword' in sentence.lower())

        num_uppercase = len(re.findall(r'[A-Z]', sentence))
        num_lowercase = len(re.findall(r'[a-z]', sentence))
        num_punctuation = len(re.findall(r'[^\w\s]', sentence))

        features.append([sentence_length, num_words, avg_word_length,
                         num_characters, num_digits, num_special_chars, contains_keyword,
                         num_uppercase, num_lowercase, num_punctuation])

    return features

def select_sentences(sentences, scores, num_sentences):
    scores = np.array(scores)
    indices = np.argsort(scores)[::-1][:num_sentences]
    ranked_sentences = [(sentences[i], scores[i]) for i in indices]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1], reverse=True)
    selected_sentences = [sentence for sentence, score in ranked_sentences]
    return selected_sentences

def generate_summary(sentences):
    summary = ' '.join(sentences)
    return summary

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['input_text']
        num_sentences = int(request.form['num_sentences'])

        preprocessed_sentences = preprocess(text)
        features = extract_features(preprocessed_sentences)

        word_counter = Counter(" ".join(preprocessed_sentences).split())
        most_common_word = word_counter.most_common(1)[0][0]

        train_sentences = [sentence for sentence in preprocessed_sentences if most_common_word in sentence]
        train_scores = [word_counter[most_common_word]] * len(train_sentences)

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_sentences)

        regression_model = LinearRegression()
        regression_model.coef_ = np.zeros(X_train.shape[1])
        regression_model.intercept_ = 0

        X_test = vectorizer.transform(preprocessed_sentences).toarray()
        predicted_scores = np.dot(X_test, regression_model.coef_) + regression_model.intercept_

        summary_sentences = select_sentences(preprocessed_sentences, predicted_scores, num_sentences=num_sentences)
        summary = generate_summary(summary_sentences)

        return render_template('index.html', input_text=text, num_sentences=num_sentences, summary=summary)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

