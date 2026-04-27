from transformers import pipeline

# Load pretrained sentiment model
classifier = pipeline("sentiment-analysis")

def bert_predict(text):
    result = classifier(text)[0]
    
    label = result['label']
    score = result['score']
    
    return label, score