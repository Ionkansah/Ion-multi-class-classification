import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk  # Import nltk

# Download NLTK data (punkt) -  Crucial for deployment!
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.warning("Downloading necessary NLTK data (punkt)... This might take a moment on first run.")
    nltk.download('punkt')
    st.success("NLTK data (punkt) downloaded successfully!")

try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    st.warning("Downloading necessary NLTK data (punkt_tab)... This might take a moment on first run.")
    nltk.download('punkt_tab')
    st.success("NLTK data (punkt_tab) downloaded successfully!")


# Load the model and vectorizer
model_path = "C:\\Users\\HP\\Documents\\MACHINE LEARNING CLASSIFICATION OF TWEETS-BI95CZ\\notebooks\\bi95cz_tweet_classification_model.joblib"
vectorizer_path = "C:\\Users\\HP\\Documents\\MACHINE LEARNING CLASSIFICATION OF TWEETS-BI95CZ\\notebooks\\bi95cz_tweet_classification_vectorizer.joblib"

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error(f"Error: Model or vectorizer file not found at the specified paths.\nModel Path: {model_path}\nVectorizer Path: {vectorizer_path}\nPlease ensure the paths are correct.")
    st.stop()


label_encoder_classes = ['Arts & Culture', 'Business & Entrepreneurship', 'Pop Culture', 'Daily Life', 'Sports & Gaming', 'Science & Technology']


def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    return ' '.join(tokens)


st.title('Tweet Classification App')

# Single Tweet Prediction
st.subheader('Predict the class of a single tweet')
single_tweet = st.text_area('Enter your tweet here:')
if st.button('Predict Single Tweet'):
    if single_tweet:
        cleaned_tweet = clean_text(single_tweet)
        vectorized_tweet = vectorizer.transform([cleaned_tweet])
        prediction = model.predict(vectorized_tweet)[0]
        st.write(f'Predicted Class: **{label_encoder_classes[prediction]}**')
    else:
        st.warning('Please enter a tweet.')

st.markdown('---')

# Batch Prediction via File Upload
st.subheader('Predict the class of multiple tweets from a CSV file')
uploaded_file = st.file_uploader(
    "Upload a CSV file containing tweets (one tweet per row in a column named 'text')", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            tweets = df['text'].tolist()
            cleaned_tweets = [clean_text(tweet) for tweet in tweets]
            vectorized_tweets = vectorizer.transform(cleaned_tweets)
            predictions = model.predict(vectorized_tweets)
            df['predicted_class'] = [label_encoder_classes[p] for p in predictions]
            st.subheader('Predictions:')
            st.dataframe(df)
            st.success('Predictions completed successfully!')
        else:
            st.error("The CSV file must contain a column named 'text'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")