import streamlit as st
import joblib

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

# Load saved model & vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")
model = joblib.load("model/model.pkl")

st.title("📧 Email Spam Classifier")
st.write("Type an email message below and this app will predict whether it is **Spam** or **Not Spam**.")

text = st.text_area("Email content", height=180, placeholder="Paste or type the email text here...")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()

        if pred == 1:
            st.error(f"🚫 SPAM (confidence: {prob:.2f})")
        else:
            st.success(f"✅ NOT SPAM (confidence: {prob:.2f})")
