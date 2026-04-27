import pandas as pd
import pickle
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
dataset = pd.read_csv("data/Restaurant_Reviews.tsv", sep='\t')

# Preprocess
corpus = [clean_text(review) for review in dataset['Review']]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = vectorizer.fit_transform(corpus).toarray()
y = dataset['Liked']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained & saved!")