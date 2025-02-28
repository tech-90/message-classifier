import streamlit as st
import pickle
import string
import nltk
import os
import urllib.request
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Ensure a writable directory
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DIR)

# Download and extract 'punkt' tokenizer manually
PUNKT_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"
punkt_path = os.path.join(NLTK_DIR, "tokenizers", "punkt.zip")

if not os.path.exists(os.path.join(NLTK_DIR, "tokenizers", "punkt")):
    print("Downloading punkt.zip manually...")
    urllib.request.urlretrieve(PUNKT_URL, punkt_path)

    # Extract punkt.zip
    with zipfile.ZipFile(punkt_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(NLTK_DIR, "tokenizers"))

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
try:
    BASE_DIR = os.getcwd()
    tfidf = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))
    model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.header("Spam" if result == 1 else "Not Spam")
