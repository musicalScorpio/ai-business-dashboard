"""
@author Sam Mukherjee
AI Assisted
Example of Hugging Face's Transformers library to summarize text and analyze sentiment
"""
import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from pdfminer.high_level import extract_text
import seaborn as sns


# Load AI models
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI setup
st.set_page_config(page_title="AI Business Insights Dashboard", layout="wide")
st.title("📊 AI-Powered Business Insights Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])


def load_csv(file):
    df = pd.read_csv(file)
    return df

def load_pdf(file):
    text = extract_text(file)
    return text


def summarize_text(text, chunk_size=500):
    # Split text into chunks of 'chunk_size'
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    return " ".join(summaries)  # Combine chunked summaries into a single summary

def analyze_sentiment(text, chunk_size=500):
    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sentiments = []
    for chunk in chunks:
        sentiment = sentiment_analyzer(chunk)[0]
        sentiments.append((sentiment['label'], sentiment['score']))

    # Aggregate sentiment scores
    positive_score = sum(score for label, score in sentiments if label == 'POSITIVE')
    negative_score = sum(score for label, score in sentiments if label == 'NEGATIVE')
    neutral_score = sum(score for label, score in sentiments if label == 'NEUTRAL')

    # Determine overall sentiment
    if positive_score > negative_score and positive_score > neutral_score:
        overall_sentiment = "POSITIVE"
    elif negative_score > positive_score and negative_score > neutral_score:
        overall_sentiment = "NEGATIVE"
    else:
        overall_sentiment = "NEUTRAL"

    confidence = max(positive_score, negative_score, neutral_score) / len(sentiments)

    return overall_sentiment, confidence


if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "pdf":
        text = load_pdf(uploaded_file)

        st.subheader("📌 AI-Powered Insights from PDF")
        summary = summarize_text(text)
        sentiment = analyze_sentiment(text)
        #Summary of summaries
        st.write("### Summary:")
        st.info(summary)

        st.write("### Sentiment Analysis:")
        if 'NEGATIVE'== str(sentiment[0]).upper():
            st.error(sentiment)
        else:
            st.success(sentiment)
    else:
        st.error("Unsupported file format. Only supporting PDF file.")
