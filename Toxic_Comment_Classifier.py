#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# # Toxic Comment Classifier

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ### Load Dataset

# In[3]:


df = pd.read_csv('/content/train.csv')
df.head()


# In[4]:


# Info of dataset
print("No of records - ", df.shape[0])
print()
df.info()


# ### Data Analysis

# In[5]:


# check for rows without any lable (+ve comments)
no_label_count = 0
rowSum = df.iloc[:, 2:].sum(axis = 1)

for i in rowSum:
  if i == 0:
    no_label_count += 1

print("Total number of rows = ", df.shape[0])
print("No. of rows without lable(+ve comments) = ", no_label_count)


# In[6]:


# Plot
x = df.iloc[:,2:].sum()
y = df.columns[2:]

plt.figure(figsize = (10, 5))
sns.barplot(x = x,y = y, alpha = 0.8, palette= ['blue', 'orange', 'red', 'green', 'violet', 'brown'])
plt.title("No. of comments per class")
plt.xlabel("No. of comments")
plt.ylabel("Class")
plt.show()


# ### Data Preprocessing

# In[7]:


# check for null values and remove unwanted data
no_of_NaN = df.isnull().sum()
print(no_of_NaN)

df.drop(['id'], axis = 1, inplace = True)
df.dropna(inplace = True)


# In[8]:


# Text cleaning and Stopword removal
stopwords = stopwords.words('english')

def clean_text(text):
  text = text.lower()
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "can not ", text)
  text = re.sub(r"don't","do not", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\W", " ", text)
  text = re.sub(r"\s+", " ", text)
  return text


def remove_stopwords(text):
  no_stop_words = []

  for w in text.split():
    if w not in stopwords:
      no_stop_words.append(w)

  return ' '.join(no_stop_words)



# In[9]:


# Stemming
stemmer = SnowballStemmer('english')

def stemming(text):
  stemmed_words = []

  for w in text.split():
    w = stemmer.stem(w)
    stemmed_words.append(w)

  return ' '.join(stemmed_words)


# In[10]:


# Preprocess the comments
df['comment_text'] = df['comment_text'].apply(clean_text)
df['comment_text'] = df['comment_text'].apply(remove_stopwords)
df['comment_text'] = df['comment_text'].apply(stemming)

df['comment_text'][:5]


# ### Model Training

# In[47]:


X = df["comment_text"]
Y = df.drop(['comment_text'], axis = 1)

# split into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[48]:


def run_pipeline(model, x_train, x_test, y_train, y_test):

  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  # check accuracy
  print("Accuracy Score = ", accuracy_score(y_test, y_pred))
  print()
  print("Classification Report :-\n", classification_report(y_test, y_pred))


# In[49]:


# Make Pipelines
LR_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(stop_words = 'english')),
                        ('lr_model', OneVsRestClassifier(LogisticRegression(), n_jobs = -1))
])


# In[50]:


run_pipeline(LR_pipeline, X_train, X_test, Y_train, Y_test)


# ### Model Testing and Results

# In[63]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def prediction(comment):
    ori_text = comment
    comment = clean_text(comment)
    comment = remove_stopwords(comment)
    comment = stemming(comment)

    predicted_values = LR_pipeline.predict([comment])[0]

    predicted_classes = [labels[i] for i in range(6) if predicted_values[i] == 1]

    if predicted_classes:
        new_text = "No need to detoxify this world shall know toxicity"
        return predicted_classes, new_text
    else:
        new_text = "This comment is Non-Toxic :)"
        return ["Non-Toxic"], new_text


# In[60]:


# Change comment and see results

# Some example test cases for each catagory
toxic = "You're so stupid, I can't stand you!"
severe_toxic = "Don't fuck around here"
obscene = "Stupid piece of shit"
threat = "I will kill you"
insult = "He is such a loser, no one wants to be around him"
identity_hate = "You are gay?"

# Use above examples of any sentence of your choice :)

comment =insult

classification_results, detoxified_text =prediction(comment)
classification_results
detoxified_text


# In[84]:


get_ipython().system('jupyter nbconvert --to script "/content/drive/MyDrive/Colab Notebooks/Toxic_Comment_Classifier.ipynb"')


# In[81]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nfrom Toxic_Comment_Classifier import prediction\nprint("Import successful!")\n\n\nfrom Toxic_Comment_Classifier import prediction\n\n# Streamlit UI\nst.title("Multi-Label Toxic Comment Classifier with Detoxification")\nst.write("Enter a comment, and the app will classify it into multiple categories and provide a detoxified version.")\n\n# Input text area\ninput_text = st.text_area("Enter your comment:", height=100, placeholder="Type your comment here...")\n\nif st.button("Classify and Detoxify"):\n    if not input_text.strip():\n        st.warning("Please enter a valid comment!")\n    else:\n        # Call the prediction function\n        predicted_classes, detoxified_text = prediction(input_text)\n\n        # Display classification results\n        st.subheader("Classification Results:")\n        if predicted_classes == ["Non-Toxic"]:\n            st.write("This comment is classified as **Non-Toxic**.")\n        else:\n            st.write("The comment is classified into the following categories:")\n            for label in predicted_classes:\n                st.write(f"- **{label.capitalize()}**")\n\n        # Display detoxified text\n        st.subheader("Detoxified Comment:")\n        st.success(detoxified_text)\n')


# In[82]:


from pyngrok import ngrok

# List active tunnels
tunnels = ngrok.get_tunnels()

# Print the public URL and tunnel name (ID)
for tunnel in tunnels:
    print(f"Public URL: {tunnel.public_url}, Tunnel Name: {tunnel.name}")

# Example of how to disconnect a specific tunnel by its public URL
tunnel_to_disconnect = 'https://7e8f-34-16-160-165.ngrok-free.app'  # Replace with the actual public URL
ngrok.disconnect(tunnel_to_disconnect)


