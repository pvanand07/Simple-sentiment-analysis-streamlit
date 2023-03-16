#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */app.py
#
# PROGRAMMER: Anand Siva P V
# DATE CREATED: 16-03-2023
# REVISED DATE: 16-03-2023
# PURPOSE: Analyze the sentiment of text input by a user

# Define Imports
import streamlit as st
from transformers import pipeline

# create sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# set up Streamlit app
st.title("Simple Sentiment Analysis App")

# get user input
user_input = st.text_input("Enter some text:")

# perform sentiment analysis when user clicks "Analyze" button
if st.button("Analyze"):
    # run sentiment analysis pipeline on user input
    result = sentiment_analysis(user_input)
    # display sentiment and confidence score
    sentiment = result[0]["label"]
    score = result[0]["score"]
    st.write("Sentiment:", sentiment)
    st.write("Confidence score:", score)
