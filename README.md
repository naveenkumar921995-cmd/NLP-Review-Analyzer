# 🤖 AI Review Analyzer Dashboard

An end-to-end NLP application that analyzes text using Machine Learning and BERT to generate sentiment, summaries, keywords, and interactive analytics in a dashboard.

---

## 🚀 Overview

This project is a complete **Natural Language Processing (NLP) pipeline + dashboard application**.

It allows users to input any text (reviews, feedback, paragraphs) and get:

- Sentiment Analysis (Positive / Negative)
- Confidence Score
- Automatic Summary
- Keyword Extraction
- Interactive Visualizations

The app supports both:
- Traditional Machine Learning (TF-IDF + Naive Bayes)
- Advanced Transformer Model (BERT)

---

## ✨ Key Features

### 🧠 NLP Features
- Text preprocessing (cleaning, stemming, stopword removal)
- TF-IDF feature extraction
- Sentiment classification using Naive Bayes
- BERT-based sentiment prediction (Hugging Face Transformers)
- Extractive text summarization using spaCy
- Keyword extraction (NOUN + PROPN)

---

### 📊 Dashboard & Analytics
- Sentiment probability chart (interactive)
- Word frequency analysis
- Model performance comparison
- Clean dashboard layout using Streamlit (cards + columns)

---

### 🎯 User Experience
- Simple and intuitive UI
- Example input button
- Guided instructions
- Fast real-time analysis

---

## 🏗️ Architecture
User Input
↓
Text Preprocessing (NLTK)
↓
TF-IDF Vectorization
↓
Model Selection
├── Naive Bayes (ML)
└── BERT (Transformers)
↓
Prediction (Sentiment + Confidence)
↓
spaCy Processing
├── Summary
└── Keywords
↓
Visualization (Plotly Dashboard)

---

## 🛠️ Tech Stack

### Frontend
- Streamlit (dashboard UI)
- Plotly (interactive charts)

### Backend
- Python
- Scikit-learn
- spaCy
- NLTK
- Transformers (Hugging Face)

### Machine Learning
- TF-IDF Vectorizer
- Naive Bayes Classifier
- BERT (pretrained transformer model)

---

## 📁 Project Structure
NLP-Review-Analyzer/
│
├── app.py # Main dashboard app
├── train_model.py # ML training script
├── preprocessing.py # Text cleaning
├── summarizer.py # Text summarization
├── keyword_extractor.py # Keyword extraction
├── bert_model.py # BERT logic
│
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
│
├── requirements.txt
├── README.md
│
└── data/
└── Restaurant_Reviews.tsv
