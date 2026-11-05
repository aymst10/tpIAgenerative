import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG demo", layout="wide")
st.title("Moustaghfir Ahmed Yassine - RAG demo")

uploaded = st.file_uploader("Chargez un fichier texte", type=["txt"])
question = st.text_input("Posez votre question")

if uploaded and question:
    text = uploaded.read().decode("utf-8", errors="ignore")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks:
        chunks = [text]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()
    best = scores.argsort()[::-1][:3]
    context = "\n---\n".join(chunks[i] for i in best)

    st.subheader("Contexte retrouvé")
    st.write(context)

    st.subheader("Réponse locale")
    st.write("Réponse basée sur le contexte ci-dessus. Branchez ici un LLM pour générer une réponse naturelle.")
