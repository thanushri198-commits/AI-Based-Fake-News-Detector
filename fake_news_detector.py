import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 AI Based Fake News Detector")
st.write("Enter news text to check whether it is REAL or FAKE.")

# Load dataset
data = pd.read_csv("fake_news.csv")

X = data["text"]
y = data["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

st.divider()

# User Input
user_input = st.text_area("Enter News Text Here")

if st.button("Detect News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector).max()

        if prediction == "REAL":
            st.success(f"🟢 This News is REAL")
        else:
            st.error(f"🔴 This News is FAKE")

        st.write(f"Confidence Score: {probability*100:.2f}%")

        # Confidence Meter
        st.progress(float(probability))