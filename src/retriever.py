from reranker import BM25
from langchain_core.documents import Document

def retrieve(query, 
             db,
             ret_trust=0.3, 
             k_doc_orig=70, 
             k_doc_final=20, 
             nlp=None,
             k1=1.5,
             b=0.75):
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_doc_orig},
    )

    docs_orig = [doc.page_content for doc in retriever.invoke(query)]

    scores = BM25(query, docs_orig[int(k_doc_final * ret_trust):], nlp=nlp, k1=k1, b=b)

    docs_bm25 = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    result = []
    for text in docs_orig[:int(k_doc_final * ret_trust)] + docs_bm25[:k_doc_final-int(k_doc_final * ret_trust)]:
        if isinstance(text, str):
            result.append(text)
        elif isinstance(text, Document):
            result.append(text.page_content)

    return result