#Installations
#Installation
#pip install streamlit transformers
#pip install scipy
#pip install transformers[torch] --upgrade

#Importations
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import streamlit as st

# Requirements
model_path = f"MarthaK-Coder/test_trainer"
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    
    # Format output dict of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }
    
    return scores

# Streamlit app
st.set_page_config(page_title="Sentiment Analysis App", page_icon=":chart_with_upwards_trend:")

# Create a navigation bar with a "Home" icon and an "About" section
st.sidebar.write(":chart_with_upwards_trend: Home")
st.sidebar.title("Navigation")
if st.sidebar.button("About"):
    st.sidebar.write("Sentiment Analysis for Covid Responses")
    st.sidebar.write("This app analyzes sentiment in text related to COVID.")

st.title("Sentiment Analysis App")

user_input = st.text_area("Copy and Paste and/or Enter a Covid tweet:")

if st.button("Analyze Sentiment"):
    if "covid" in user_input.lower():
        scores = sentiment_analysis(user_input)
        st.write("Output:")
        for label, score in scores.items():
            st.write(f"{label}: {score:.3f}")
    else:
        st.warning("Please enter a valid text related to COVID.") 

