import nltk
nltk.download('stopwords')
import streamlit as st
import pickle
from collections import Counter

import plotly.graph_objects as go
import plotly.express as px

from preprocessing import clean_text
from summarizer import summarize_text
from keyword_extractor import extract_keywords
from bert_model import bert_predict

# =========================
# 🔹 LOAD MODELS
# =========================
import streamlit as st
import pickle

@st.cache_resource
def load_ml_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_ml_model()
# =========================
# 🔹 PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Review Analyzer", layout="wide")

# =========================
# 🔹 TITLE & DESCRIPTION
# =========================
st.title("🤖 AI Review Analyzer Dashboard")

st.markdown("""
### 🔍 What this app does:
Analyze any text and get:
- ✅ Sentiment Analysis  
- ✂️ Automatic Summary  
- 🔑 Keyword Extraction  
- 📊 Interactive Visualizations  

### ✍️ What to enter:
- Reviews (restaurant, product, service)
- Customer feedback
- Any paragraph of text
""")

# =========================
# 🔹 SAMPLE INPUT
# =========================
if st.button("✨ Use Sample Review"):
    st.session_state["user_input"] = "The food was amazing but the service was slow. Overall, I liked the ambiance and would visit again."

# =========================
# 🔹 INPUT
# =========================
user_input = st.text_area(
    "Enter your text:",
    value=st.session_state.get("user_input", ""),
    placeholder="Example: The food was great but service was slow..."
)

st.info("💡 Tip: Write at least 1–2 sentences for better insights.")

# =========================
# 🔹 MODEL SELECT
# =========================
model_choice = st.selectbox(
    "Choose Model",
    ["Machine Learning", "BERT (Advanced)"]
)

st.divider()

# =========================
# 🔹 ANALYZE
# =========================
if st.button("🚀 Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter text to analyze.")
    else:

        # =========================
        # 🔹 PREDICTION
        # =========================
        if model_choice == "Machine Learning":
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned]).toarray()

            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
            confidence = max(prob)

        else:
            label, score = bert_predict(user_input)

            sentiment = label
            confidence = score
            prob = [1 - score, score]

        # =========================
        # 🔹 DASHBOARD CARDS
        # =========================
        st.markdown("## 📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sentiment", sentiment)

        with col2:
            st.metric("Confidence", f"{confidence:.2f}")

        with col3:
            st.metric("Model Used", model_choice)

        # =========================
        # 📊 SENTIMENT CHART
        # =========================
        st.subheader("📊 Sentiment Probability")

        fig = go.Figure(data=[
            go.Bar(x=["Negative", "Positive"], y=prob)
        ])
        fig.update_layout(title="Sentiment Score", xaxis_title="Class", yaxis_title="Probability")

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # ✂️ SUMMARY + KEYWORDS
        # =========================
        col4, col5 = st.columns(2)

        with col4:
            st.subheader("✂️ Summary")
            summary = summarize_text(user_input)
            st.write(summary)

        with col5:
            st.subheader("🔑 Keywords")
            keywords = extract_keywords(user_input)
            st.write(", ".join(keywords))

        # =========================
        # 📈 WORD FREQUENCY
        # =========================
        st.subheader("📈 Word Frequency")

        words = user_input.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)

        words_list = [w[0] for w in common_words]
        counts = [w[1] for w in common_words]

        fig2 = px.bar(
            x=words_list,
            y=counts,
            labels={'x': 'Words', 'y': 'Count'},
            title="Top Words"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # 📊 MODEL COMPARISON
        # =========================
        st.subheader("📊 Model Performance Comparison")

        models = ["LogReg", "NaiveBayes", "SVM", "RandomForest"]
        accuracy = [0.80, 0.81, 0.78, 0.75]

        fig3 = px.bar(
            x=models,
            y=accuracy,
            title="Model Accuracy"
        )

        st.plotly_chart(fig3, use_container_width=True)
