import spacy
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

# ✅ Load model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
def summarize_text(text, ratio=0.3):
    doc = nlp(text)
    
    word_freq = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text not in punctuation:
            word_freq[word.text] = word_freq.get(word.text, 0) + 1

    max_freq = max(word_freq.values()) if word_freq else 1

    for word in word_freq:
        word_freq[word] /= max_freq

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.text]

    select_len = int(len(list(doc.sents)) * ratio)
    summary = nlargest(select_len, sentence_scores, key=sentence_scores.get)

    return " ".join([sent.text for sent in summary])
