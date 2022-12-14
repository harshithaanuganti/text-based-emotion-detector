#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Harshitha Anuganti
"""

import streamlit as st
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy

st.title("Text-based Emotion Recognition")
st.markdown("Harshitha Anuganti | MSBA 6490 | v1.0")
st.markdown("Sentiment analysis is a method to identify people’s attitudes, sentiments, and emotions towards a given goal, such as people, activities, organizations, services, subjects, and products. Emotion detection is a subset of sentiment analysis as it predicts the unique emotion rather than just stating positive, negative, or neutral.")
st.markdown("This application helps the user identify the emotion behind the text. This model identifies 13 different emotions, namely, happiness, sadness, hate, love, fun, boredom, anger, relief, worry, surprise, enthusiasm, empty and neutral")
st.markdown("How to use? The user can enter the text in the text box and press enter. Then the emotion behind the text is identified and displayed. A gif of the emotion is also displayed for enhancing user-experience.")
# st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)

def load_model():
    with open("model.pkl", "rb") as file1:
        model = pkl.load(file1)
    return model

def load_model_rf():
    with open("model_rf.pkl", "rb") as file1:
        model = pkl.load(file1)
    return model

def load_sentiment_labels():
    with open("sentiment_labels.pkl", "rb") as file1:
        sentiment_labels = pkl.load(file1)
    return sentiment_labels

def load_tfidfvectorizer_model():
    with open("vectorizer.pkl", "rb") as file1:
        TfidfVectorizer_model = pkl.load(file1)
    return TfidfVectorizer_model


def run():
    query = st.text_input("Enter the text here:", "What a lovely day!!")
    output_label = int(model.predict(vectorizer.transform([query]).toarray())[0])
    st.markdown("Emotion:")
    if output_label == 0:
        st.markdown("Empty")
        st.markdown("![Alt Text](https://tenor.com/view/feeling-nothing-feeling-empty-bored-kid-stroty-of-my-life-gif-17013742)", unsafe_allow_html=True)
    elif output_label == 1:
        st.markdown("Sadness")
        st.markdown("![Alt Text](https://media.giphy.com/media/13wRu1xRTswrcc/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 2:
        st.markdown("Enthusiasm")
        st.markdown("![Alt Text](https://media.giphy.com/media/l0IylDbgZ9UBb0Boc/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 3:
        st.markdown("Neutral")
        st.markdown("![Alt Text](https://media.giphy.com/media/NUerTUMGyYyKoUl0pK/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 4:
        st.markdown("Worry")
        st.markdown("![Alt Text](https://media.giphy.com/media/BFg8adASnmMrS/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 5:
        st.markdown("Surprise")
        st.markdown("![Alt Text](https://media.giphy.com/media/5p2wQFyu8GsFO/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 6:
        st.markdown("Love")
        st.markdown("![Alt Text](https://media.giphy.com/media/BNb0VikAaRwmUvsEnY/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 7:
        st.markdown("Fun")
        st.markdown("![Alt Text](https://media.giphy.com/media/w5eFyOHmkS8uc/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 8:
        st.markdown("Hate")
        st.markdown("![Alt Text](https://media.giphy.com/media/l1J9u3TZfpmeDLkD6/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 9:
        st.markdown("Happiness")
        st.markdown("![Alt Text](https://media.giphy.com/media/8i7IQbqY4iXuD3MDRT/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 10:
        st.markdown("Boredom")
        st.markdown("![Alt Text](https://media.giphy.com/media/Dem7W3KWljaFbaNgR5/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 11:
        st.markdown("Relief")
        st.markdown("![Alt Text](https://media.giphy.com/media/4PT6v3PQKG6Yg/giphy.gif)", unsafe_allow_html=True)
    elif output_label == 12:
        st.markdown("Anger")
        st.markdown("![Alt Text](https://media.giphy.com/media/icSfwHwd8ykus/giphy.gif)", unsafe_allow_html=True)
    else:
        st.markdown("invalid")

    st.markdown("The data is taken from kaggle which is originally taken from a public domain platform, data.world. The dataset has 40k records and 13 different emotions(classes). The models that are used are Gaussian Naive Bayes and Random Forest.")

    
model = load_model()
# model_rf = load_model_rf()
vectorizer = load_tfidfvectorizer_model()
sentiment_labels = load_sentiment_labels


if __name__ == '__main__':
    run()
