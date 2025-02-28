import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Ensure a writable directory
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")

# Force download punkt correctly
nltk.data.path.append(NLTK_DIR)  # Set download directory first
nltk.download('punkt', download_dir=NLTK_DIR)
nltk.download('stopwords', download_dir=NLTK_DIR)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)  # Tokenizing safely

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load model & vectorizer using relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    st.header("Spam" if result == 1 else "Not Spam")
