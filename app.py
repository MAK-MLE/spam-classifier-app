import streamlit as st
import joblib
import string
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# 📥 Download stopwords only if not already present
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop_words = set(stopwords.words('english'))

# 🧠 Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
stemmer = PorterStemmer()

# 🔍 Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    filtered = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(filtered)

# 🌐 Streamlit UI setup
st.set_page_config(page_title="📧 Email Spam Classifier", layout="centered")
st.title("📧 Email Spam Classifier")

if "history" not in st.session_state:
    st.session_state.history = []

email_text = st.text_area("✉️ Enter email content here:", height=200)
threshold = st.slider("🔧 Spam Detection Threshold", 0.0, 1.0, 0.5, 0.01)

col1, col2 = st.columns([1, 1])
with col1:
    classify = st.button("🔍 Classify")
with col2:
    reset = st.button("🔄 Reset")

# 🧪 Classify Email
if classify:
    if not email_text.strip():
        st.warning("Please enter an email message.")
    else:
        cleaned = preprocess(email_text)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0][1]  # Probability it's spam
        result = int(proba > threshold)
        label = "⚠️ SPAM" if result == 1 else "✅ NOT SPAM"

        st.markdown(f"### Prediction: {label}")
        st.progress(int(proba * 100))
        st.write(f"🧪 Spam Probability: `{proba:.2f}` | Threshold: `{threshold:.2f}`")
        st.text_area("🔍 Preprocessed Text", cleaned, height=100)
        st.session_state.history.append((email_text, label, f"{proba:.2f}"))

# 🔁 Reset history
if reset:
    st.session_state.history = []
    st.rerun()

# 🕓 Conversation History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Conversation History")
    for i, (msg, label, prob) in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{i}. {label} (Confidence: {prob})"):
            st.markdown(msg)

# 🧪 Optional test example
if st.button("🧪 Run Spam Test Example"):
    sample = "Congratulations! You've won a free iPhone. Click here to claim your prize."
    cleaned = preprocess(sample)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0][1]
    result = int(proba > threshold)
    label = "⚠️ SPAM" if result == 1 else "✅ NOT SPAM"

    st.markdown(f"**Test Result:** {label} (Confidence: `{proba:.2f}`)")
    st.text_area("Sample Email", sample)
    st.text_area("Cleaned Text", cleaned)
