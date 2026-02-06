from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_knowledge(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    return chunks


def build_retriever(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


def retrieve_evidence(query, vectorizer, vectors, chunks, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]
