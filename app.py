import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# ---------- Preprocessing ----------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---------- Load Model & Vectorizer ----------
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------- Streamlit UI ----------
st.title("📰 Fake News Detection App")

input_news = st.text_area("Enter the news content:")

if st.button("Predict"):
    # 1. Preprocess
    transformed_news = transform_text(input_news)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_news])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.success("✅ This news is REAL.")
    else:
        st.error("❌ This news is FAKE.")