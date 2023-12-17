from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model_cnn_bilstm2.h5')

# Load the Word2Vec model
word2vec_model = Word2Vec.load('word2vec_model.bin')
word_vectors = word2vec_model.wv

# Preprocessing functions
def preprocess_text(text):
    # Case Folding
    text = text.lower()

    # Remove Punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Remove Numbers and Special Characters
    text = re.sub('[^A-Za-z ]+', '', text)
    text = re.sub(r'²', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"\W", " ", text)

    # Remove Whitespace
    text = re.sub(r"\s+", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove Stopwords
    factory = StopWordRemoverFactory()
    stop_words = factory.get_stop_words() 
    stop_words.extend(['aaaaa', 'aaaa', 'aama','yaha','nya','pke', 'dehh', 'wkwk','wkwkwk', 'dpt', 'jg', 'yg', 'yahahahahaha','hahahaha', 'hahaha', 'haha', 'hehe','km','iya', 'msih', 'jga', 'ddpn', 'dsruh', 'dlu', 'ng', 'sih', 'lah','the','loh', 'kok', 'wes', 'knp', 'bro', 'gaes', 'guys', 'ges', 'iye', 'ct', 'heh'])
    stop_words = set(stop_words)
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text

# Get sentiment label
def get_sentiment_label(sentiment):
    if sentiment == 0:
        return 'Negatif'
    elif sentiment == 1:
        return 'Netral'
    elif sentiment == 2:
        return 'Positif'
    else:
        return 'Tidak Diketahui'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_text = preprocess_text(text)

    # Convert text to sequence of word vectors
    seq = [word_vectors.key_to_index[word] for word in preprocessed_text.split() if word in word_vectors.key_to_index]
    padded_seq = pad_sequences([seq], maxlen=64)

    sentiment = np.argmax(model.predict(padded_seq), axis=1)
    sentiment_label = get_sentiment_label(sentiment[0])
    return render_template('index.html', text=text, sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
