import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords once
nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function (same as used in training)
def preprocess(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    filtered = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="📧 Email Spam Classifier", layout="centered")
st.title("📧 Email Spam Classifier")

if "history" not in st.session_state:
    st.session_state.history = []

email_text = st.text_area("✉️ Enter email content here:", height=200)

col1, col2 = st.columns([1, 1])

with col1:
    classify = st.button("🔍 Classify")
with col2:
    reset = st.button("🔄 Reset")

if classify:
    if not email_text.strip():
        st.warning("Please enter an email message.")
    else:
        cleaned = preprocess(email_text)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]  # Probability of spam

        label = "⚠️ SPAM" if result == 1 else "✅ NOT SPAM"
        st.markdown(f"### Prediction: {label}")
        st.progress(int(proba * 100))
        st.write(f"🧪 Spam Probability: `{proba:.2f}`")
        st.text_area("🔍 Preprocessed Text", cleaned, height=100)
        st.session_state.history.append((email_text, label, f"{proba:.2f}"))

if reset:
    st.session_state.history = []
    st.rerun()


if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Conversation History")
    for i, (msg, label, prob) in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{i}. {label} (Confidence: {prob})"):
            st.markdown(msg)

# Optional test example
if st.button("🧪 Run Spam Test Example"):
    sample = "Congratulations! You've won a free iPhone. Click here to claim your prize."
    cleaned = preprocess(sample)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]
    label = "⚠️ SPAM" if result == 1 else "✅ NOT SPAM"
    st.markdown(f"**Test Result:** {label} (Confidence: `{proba:.2f}`)")
    st.text_area("Sample Email", sample)
    st.text_area("Cleaned", cleaned)
