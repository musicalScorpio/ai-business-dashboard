"""
@author Sam Mukherjee
AI Assisted
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load AI models
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI setup
st.set_page_config(page_title="AI Business Insights Dashboard", layout="wide")
st.title("📊 AI-Powered Business Insights Dashboard")

# Upload Public Company financial info
uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])


def load_csv(file):
    df = pd.read_csv(file)
    return df


def summarize_text(text):
    return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']


def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]


if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "csv":
        df = load_csv(uploaded_file)
        st.dataframe(df)

        # Generate insights
        st.subheader("📌 AI-Powered Insights")
        sample_text = " ".join(df.iloc[:, 0].astype(str).tolist()[:5])  # Take sample text from first column

        summary = summarize_text(sample_text)
        sentiment = analyze_sentiment(sample_text)

        st.write("### Summary:")
        st.info(summary)

        st.write("### Sentiment Analysis:")
        st.success(f"Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")

        # Sample visualization
        st.subheader("📊 Data Overview")
        st.line_chart(df.select_dtypes(include=["number"]))

    else:
        st.error("PDF support coming soon! Upload a CSV file for now.")
