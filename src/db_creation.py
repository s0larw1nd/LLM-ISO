import os
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from scanner import CustomTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
import re

def clean_documents(documents):
    for doc in documents:
        doc.page_content = re.sub(r"(?<!\d)\t", "", doc.page_content)
        while '\n\n' in doc.page_content: doc.page_content = re.sub(r"\n\n", "\n", doc.page_content)
    return documents

def create_db(doc_path,
              persistent_directory,
              embeddings,
              text_splitter=None):
    
    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        #text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=256)
        #text_splitter = CustomTextSplitter(chunk_size=700)
    
    if os.path.isfile(doc_path):
        loader = Docx2txtLoader(doc_path)
        documents = loader.load()
        docs = text_splitter.split_documents(clean_documents(documents))
    elif os.path.isdir(doc_path):
        docx_files = [f for f in os.listdir(doc_path) if f.endswith('.docx')]
        docs = []
        for file_name in docx_files:
            file_path = os.path.join(doc_path, file_name)
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            split_docs = text_splitter.split_documents(clean_documents(documents))
            docs.extend(split_docs)
    else:
        raise ValueError("Некорректный путь")

    return Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

if __name__ == "__main__":
    import config
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    create_db(config.DEFAULT_DOC_FILE, config.DEFAULT_DB_DIR, HuggingFaceEmbeddings(model_name=config.DEFAULT_EMBEDDINGS))