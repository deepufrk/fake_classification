import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
#nltk.download('stopwords')
st.title('Email Spam Classifier')

spam_model = joblib.load('svm_model.joblib')
vectorizer = joblib.load('CountVectorizer.joblib')
inp_text = st.text_area('Paste the email to determine whether it is Spam or Ham',height=200)

vectorised_text = vectorizer.transform([inp_text])
pred = ''
# add a placeholder

def spam_predict(inp_text):
    prediction = spam_model.predict(inp_text)
    if prediction == 0:
        pred = 'Ham'
    else:
        pred = 'Spam'
    return pred
if st.button('Submit'):
    st.write('The email you entered is:',spam_predict(vectorised_text))
