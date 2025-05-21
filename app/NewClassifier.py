import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A3, landscape
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import numpy as np
from reportlab.graphics.shapes import Drawing, Circle
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import ImageReader
from altair_saver import save
import os
from sklearn.metrics import confusion_matrix

# Download NLTK data (punkt) - Required for deployment!
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
model_path = "notebooks/bi95cz_tweet_classification_model.joblib"
vectorizer_path = "notebooks/bi95cz_tweet_classification_vectorizer.joblib"

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error(f"Error: Model or vectorizer file not found at the specified paths.\nModel Path: {model_path}\nVectorizer Path: {vectorizer_path}\nPlease ensure the paths are correct.")
    st.stop()


label_encoder_classes = [
    'Arts & Culture',
    'Business & Entrepreneurship',
    'Pop Culture',
    'Daily Life',
    'Sports & Gaming'
]


def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    return ' '.join(tokens)


def safe_class_name(idx, class_names):
    try:
        return class_names[idx]
    except Exception:
        return "Unknown"


# --- Visualization Functions ---
def plot_class_distribution(predictions, class_names, title="Predicted Class Distribution"):
    """Plots the distribution of predicted classes."""
    pred_counts = pd.Series(predictions).value_counts().reset_index()
    pred_counts.columns = ['class_index', 'count']
    pred_counts['class_name'] = pred_counts['class_index'].map(lambda x: class_names[x])

    chart = alt.Chart(pred_counts).mark_bar().encode(
        x=alt.X('class_name', sort='-y'),
        y='count',
        color='class_name',
        tooltip=['class_name', 'count']
    ).properties(
        title=title
    )
    st.altair_chart(chart, use_container_width=True)


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plots a confusion matrix using matplotlib and seaborn."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    st.pyplot(fig)


def plot_class_distribution_matplotlib(data, class_names, container=None):
    """Plots the distribution of predicted classes using matplotlib/seaborn."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='predicted_class', data=data, order=data['predicted_class'].value_counts().index, ax=ax)
    plt.title("Predicted Class Distribution")
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if container is not None:
        container.pyplot(fig)
    return fig


# --- PDF Report Generation ---
def create_pdf_report(data, class_names, cm=None, filename="tweet_classification_report.pdf"):
    """Generates a PDF report containing predictions and visualizations."""
    buffer = BytesIO()  # Create an in-memory buffer for the PDF
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A3))
    doc.topMargin = 0.75 * inch
    doc.bottomMargin = 0.75 * inch
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Tweet Classification Report", styles['h1']))
    story.append(Paragraph("This report summarizes the results of the tweet classification.", styles['Normal']))
    story.append(Paragraph(" ", styles['Normal']))  # Add some space

    # Predictions Table (if applicable)
    if isinstance(data, pd.DataFrame) and not data.empty:
        story.append(Paragraph("Prediction Results:", styles['h2']))
        table_data = [data.columns.tolist()] + data.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#CCCCCC'),
            ('GRID', (0, 0), (-1, -1), 1, 'black')
        ]))
        story.append(table)
        story.append(Paragraph(" ", styles['Normal']))  # Add some space

    # Class Distribution Plot (Matplotlib/Seaborn only for PDF)
    story.append(Paragraph("Predicted Class Distribution:", styles['h2']))
    fig = plot_class_distribution_matplotlib(data, class_names)
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close(fig)
    try:
        img = Image(img_buffer, width=5 * inch, height=4 * inch)
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"Error embedding class distribution: {e}", styles['Normal']))

    # Confusion Matrix (Received data)
    if cm is not None:
        story.append(Paragraph("Confusion Matrix:", styles['h2']))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title("Confusion Matrix (Uploaded Data)")
        cm_buffer = BytesIO()
        plt.savefig(cm_buffer, format='png')
        cm_buffer.seek(0)
        try:
            img = Image(cm_buffer, width=5 * inch, height=4 * inch)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Error embedding confusion matrix: {e}", styles['Normal']))
        finally:
            plt.close(fig)
        story.append(Paragraph(" ", styles['Normal']))  # Add some space

    doc.build(story)
    buffer.seek(0)  # Reset buffer to beginning
    return buffer.getvalue()


st.title('Tweet Classification App')

# --- Sidebar for Options ---
st.sidebar.header("Options")
display_cm = st.sidebar.checkbox("Show Confusion Matrix (on Training Data)", value=False)

# Single Tweet Prediction
st.subheader('Name the class of a single tweet')
single_tweet = st.text_area('Enter your tweet here:')
if st.button('Name Single Tweet Category'):
    if single_tweet:
        cleaned_tweet = clean_text(single_tweet)
        vectorized_tweet = vectorizer.transform([cleaned_tweet])
        prediction = model.predict(vectorized_tweet)[0]
        st.write(f'Predicted Class: **{label_encoder_classes[prediction]}**')

st.markdown('---')

# Batch Prediction via File Upload
st.subheader('Name the class of multiple tweets from a CSV file')
uploaded_file = st.file_uploader(
    "Upload a CSV file containing tweets (one tweet per row in a column named 'text'). For confusion matrix, add a 'true_label' column.", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'true_label' in df.columns:
            allowed = df['true_label'].isin(label_encoder_classes)
        if not allowed.all():
            st.warning(f"Some rows have true_label(s) not in the allowed classes and will be ignored: {set(df.loc[~allowed, 'true_label'])}")
        df = df[allowed]
        if 'text' in df.columns:
            cleaned_tweets = [clean_text(tweet) for tweet in df['text'].tolist()]
            vectorized_tweets = vectorizer.transform(cleaned_tweets)
            predictions = model.predict(vectorized_tweets)
            df['predicted_class'] = [safe_class_name(p, label_encoder_classes) for p in predictions]
            st.subheader('Predictions:')
            st.dataframe(df)
            st.success('Predictions completed successfully!')

            # Filter for allowed true_label classes
            cm = None
            if 'true_label' in df.columns:
                allowed = df['true_label'].isin(label_encoder_classes)
                if not allowed.all():
                    st.warning(f"Some rows have true_label(s) not in the allowed classes and will be ignored: {set(df.loc[~allowed, 'true_label'])}")
                    df = df[allowed]
                true_indices = df['true_label'].map(lambda x: label_encoder_classes.index(x)).tolist()
                cm = confusion_matrix(true_indices, predictions[:len(true_indices)])
                st.subheader("Confusion Matrix (Uploaded Data)")
                plot_confusion_matrix(cm, label_encoder_classes)
            else:
                st.info("No 'true_label' column found in the uploaded file. Confusion matrix will not be shown.")

            # Visualizations for batch predictions (Matplotlib/Seaborn only)
            st.subheader("Batch Prediction Visualizations")
            fig = plot_class_distribution_matplotlib(df, label_encoder_classes, container=st)

            # PDF Report Generation for batch predictions (with actual cm if available)
            pdf_bytes = create_pdf_report(df, label_encoder_classes, cm=cm)
            st.download_button(
                label="Download Prediction Report (PDF)",
                data=pdf_bytes,
                file_name="tweet_classification_report.pdf",
                mime="application/pdf"
            )

        else:
            st.error("The uploaded file does not contain the expected data.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display Confusion Matrix (Optional) ---
if display_cm:
    st.subheader("Confusion Matrix (on Training Data)")
    # Load your actual confusion matrix here. For demonstration, I'm using a dummy.
    dummy_cm = np.array([[10, 2, 1, 0, 0, 0],
                         [1, 20, 3, 0, 1, 2],
                         [2, 5, 150, 10, 5, 0],
                         [0, 1, 8, 40, 2, 0],
                         [0, 0, 5, 3, 100, 1],
                         [0, 2, 0, 1, 2, 30]])
    plot_confusion_matrix(dummy_cm, label_encoder_classes)
