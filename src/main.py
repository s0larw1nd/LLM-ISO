import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from scanner import CustomTextSplitter

def answer_query(query, 
                 model_name="./models/Qwen2.5-7B-Instruct-merged",
                 embeddings_name="./models/FRIDA",
                 persistent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db"),
                 file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "doc.docx"),
                 model = None,
                 tokenizer = None):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)

    if not os.path.exists(persistent_directory):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=256)
        text_splitter = CustomTextSplitter(chunk_size=300)
        docs = text_splitter.split_documents(documents)

        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    else:
        print("БД уже существует")

    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15},
    )
    relevant_docs = retriever.invoke(query)

    #for i, doc in enumerate(relevant_docs, 1):
        #print(f"Документ {i}:\n{doc.page_content}\n")
    context_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

    if model is None: model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    if tokenizer is None: tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
    {"role": "system", "content": 
     f"""Используя только данный контекст, дай максимально подробный ответ на вопрос о содержании документа. 
     Крайне важно: ни в коем случае не добавляй ничего не из контекста. По возможности, пиши источники информации (номера и названия пунктов документа).
     Если в тексте ответа есть ссылка на какой-либо документ, по возможности приведи нужный текст из этого документа.
     Пиши всегда на русском языке. 
     КОНТЕКСТ:
     {context_docs}
     КОНЕЦ КОНТЕКСТА"""
     },
    {"role": "user", "content": "ВОПРОС: " + query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":
    print(answer_query("О чём говорится в пункте 7.1.6?"))