import math
import spacy

def lemmatize_spacy(text, nlp):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

def BM25(query, documents, k1=1.5, b=0.75, nlp = None):
    if nlp is None: nlp = spacy.load("ru_core_news_sm")

    q_terms = lemmatize_spacy(query,nlp).strip().strip("?").lower().split()
    docs = [lemmatize_spacy(doc,nlp).strip().lower().split() for doc in documents]
    avgdl = sum(len(doc) for doc in docs) / len(docs)
    
    df = {}
    for term in q_terms:
        for doc in docs:
            if term in doc:
                df[term] = df.get(term, 0) + 1

    scores = []
    for doc in docs:
        score = 0
        for term in q_terms:
            tf = doc.count(term)
            if tf == 0: continue
            idf = math.log((len(docs) - df[term] + 0.5) / (df[term] + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(score)
    
    return dict(zip(documents, scores))