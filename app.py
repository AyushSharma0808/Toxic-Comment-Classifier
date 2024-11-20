import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download necessary nltk resources
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('train.csv')  # Ensure train.csv is in the same directory as this file
df.drop(['id'], axis=1, inplace=True)
df.dropna(inplace=True)

# Preprocessing functions
stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def remove_stopwords(text):
    no_stop_words = [w for w in text.split() if w not in stopwords]
    return ' '.join(no_stop_words)

def stemming(text):
    stemmed_words = [stemmer.stem(w) for w in text.split()]
    return ' '.join(stemmed_words)

# Apply preprocessing to comments
df['comment_text'] = df['comment_text'].apply(clean_text)
df['comment_text'] = df['comment_text'].apply(remove_stopwords)
df['comment_text'] = df['comment_text'].apply(stemming)

X = df["comment_text"]
Y = df.drop(['comment_text'], axis=1)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the classification pipeline
LR_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr_model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))
])

# Train and evaluate the model
LR_pipeline.fit(X_train, Y_train)
y_pred = LR_pipeline.predict(X_test)

print("Accuracy Score = ", accuracy_score(Y_test, y_pred))
print("Classification Report :-\n", classification_report(Y_test, y_pred))

# Streamlit UI
st.title("Multi-Label Toxic Comment Classifier")
st.write("Enter a comment, and the app will classify it into multiple categories.")

# Input text area
input_text = st.text_area("Enter your comment:", height=100, placeholder="Type your comment here...")

if st.button("Classify"):
    if not input_text.strip():
        st.warning("Please enter a valid comment!")
    else:
        # Prediction function
        comment = clean_text(input_text)
        comment = remove_stopwords(comment)
        comment = stemming(comment)

        predicted_value = LR_pipeline.predict([comment])[0]
        predicted_classes = [Y.columns[i] for i in range(6) if predicted_value[i] == 1]

        st.subheader("Classification Results:")
        if not predicted_classes:
            st.write("This comment is classified as **Non-Toxic**.")
        else:
            st.write("The comment is classified into the following categories:")
            for label in predicted_classes:
                st.write(f"- **{label.capitalize()}**")
