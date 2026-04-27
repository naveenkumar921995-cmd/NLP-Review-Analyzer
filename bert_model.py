from transformers import pipeline

# ✅ Lightweight model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def bert_predict(text):
    result = classifier(text)[0]
    return result['label'], result['score']
