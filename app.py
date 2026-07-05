#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */app.py
#
# PROGRAMMER: Anand Siva P V
# DATE CREATED: 16-03-2023
# REVISED DATE: 16-03-2023
# PURPOSE: Analyze the sentiment of text input by a user

# Define Imports
import pandas as pd
import streamlit as st
from transformers import pipeline

from csv_upload import build_batch_predictions, find_text_column

# create sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# set up Streamlit app
st.title("Simple Sentiment Analysis App")

# get user input
user_input = st.text_input("Enter some text:")

# perform sentiment analysis when user clicks "Analyze" button
if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text before analyzing.")
    else:
        # run sentiment analysis pipeline on user input
        result = sentiment_analysis(user_input)
        # display sentiment and confidence score
        sentiment = result[0]["label"]
        score = result[0]["score"]
        st.write("Sentiment:", sentiment)
        st.write("Confidence score:", score)

st.divider()
st.subheader("Batch analyze CSV")
st.caption("Upload a CSV with review text, text, or Xquik Tweet Text columns.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read CSV: {error}")
    else:
        text_column = find_text_column(uploaded_df)
        if text_column is None:
            st.error("No text column found. Use review, text, tweet, feedback, or Tweet Text.")
        else:
            results_df = build_batch_predictions(uploaded_df, sentiment_analysis)
            if results_df.empty:
                st.warning("No non-empty text rows found in the selected column.")
            else:
                st.caption(f"Analyzed {len(results_df)} rows from {text_column}.")
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    results_df.to_csv(index=False),
                    "sentiment_predictions.csv",
                    "text/csv",
                )
