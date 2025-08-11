import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from datetime import datetime
import base64

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    filtered = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(filtered)

# Page config
st.set_page_config(page_title="📧 Spam Classifier", layout="wide")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "sound_trigger" not in st.session_state:
    st.session_state.sound_trigger = 0

# Toggle dark mode
st.sidebar.markdown("## ⚙️ Settings")
dark_toggle = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)

if dark_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_toggle
    st.rerun()

# Apply dark or light theme manually via custom CSS
custom_css = """
<style>
body {
    background-color: %s;
    color: %s;
}
textarea, .stTextInput > div > div > input {
    background-color: %s !important;
    color: %s !important;
    border: 1px solid %s !important;
}
.stButton button {
    background-color: #5865F2;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
}
.chat-bubble {
    margin: 10px;
    padding: 10px 15px;
    border-radius: 20px;
    max-width: 70%%;
    word-wrap: break-word;
}
.user-bubble {
    background-color: %s;
    margin-left: auto;
    text-align: right;
}
.bot-bubble {
    background-color: %s;
    margin-right: auto;
    text-align: left;
    border-left: 4px solid #00FFB3;
}
</style>
""" % (
    ("#0e1117", "#ffffff", "#1e1e2f", "#ffffff", "#444444", "#5865F2", "#2b2d42") if st.session_state.dark_mode else
    ("#e6f0ff", "#000000", "#ffffff", "#000000", "#cccccc", "#d0e0ff", "#ffffff")
)
st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💬 Spam Classifier-MAK </h1>", unsafe_allow_html=True)

email_text = st.text_area("✉️ Enter your email here:", height=150)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    classify = st.button("🚀 Classify")
with col2:
    reset = st.button("🔄 Reset")
with col3:
    export = st.button("📁 Export Chat")

# Sound (ding) on output
sound_file = "https://www.myinstants.com/media/sounds/ding-sound-effect_2.mp3"
if st.session_state.sound_trigger:
    st.markdown(f"""
    <audio autoplay>
        <source src="{sound_file}" type="audio/mpeg">
    </audio>
    """, unsafe_allow_html=True)
    st.session_state.sound_trigger = 0

if classify and email_text.strip():
    cleaned = preprocess(email_text)
    vec = vectorizer.transform([cleaned])
    threshold = 0.32  # more sensitive to spam
    proba = model.predict_proba(vec)[0][1]
    result = 1 if proba >= threshold else 0
    label = "⚠️ SPAM" if result == 1 else "✅ NOT SPAM"


    st.session_state.history.append({
        "text": email_text,
        "label": label,
        "proba": f"{proba:.2f}",
        "time": datetime.now().strftime("%H:%M:%S")
    })

    st.session_state.sound_trigger = 1
    st.rerun()

if reset:
    st.session_state.history = []
    st.rerun()

if export and st.session_state.history:
    history_txt = "\n".join([f"[{h['time']}] You: {h['text']}\nBot: {h['label']} ({h['proba']})" for h in st.session_state.history])
    b64 = base64.b64encode(history_txt.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">📥 Download Chat History</a>'
    st.markdown(href, unsafe_allow_html=True)

# Scrollable history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Chat History")
    for item in st.session_state.history[-10:]:
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            👤 {item['text']}
        </div>
        <div class="chat-bubble bot-bubble">
            🤖 {item['label']}<br><small>Confidence: {item['proba']}</small>
        </div>
        """, unsafe_allow_html=True)

