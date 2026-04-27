import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================
# 🔹 Ensure stopwords available
# =========================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =========================
# 🔹 Initialize tools
# =========================
ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

# keep "not" for sentiment meaning
if 'not' in stop_words:
    stop_words.remove('not')

# =========================
# 🔹 Text Cleaning Function
# =========================
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    # Remove non-letters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Lowercase + tokenize
    words = text.lower().split()

    # Remove stopwords + stemming
    words = [ps.stem(word) for word in words if word not in stop_words]

    return ' '.join(words)
