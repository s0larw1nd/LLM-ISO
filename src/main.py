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
                 path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "doc.docx"),
                 model = None,
                 tokenizer = None,
                 embeddings = None,
                 retriever = None,
                 text_splitter = None,
                 context_docs = [],
                 new_db = False):
    
    if context_docs == []:
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)

        if not os.path.exists(persistent_directory) or new_db:
            print("Создание БД")

            if text_splitter is None:
                #text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
                text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=256)
                #text_splitter = CustomTextSplitter(chunk_size=300)
            
            if os.path.isfile(path):
                loader = Docx2txtLoader(path)
                documents = loader.load()
                docs = text_splitter.split_documents(documents)
            elif os.path.isdir(path):
                docx_files = [f for f in os.listdir(path) if f.endswith('.docx')]
                docs = []
                for file_name in docx_files:
                    file_path = os.path.join(path, file_name)
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                    split_docs = text_splitter.split_documents(documents)
                    docs.extend(split_docs)
            else:
                raise ValueError("Некорректный путь")

            db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        else:
            print("БД уже существует")

        if retriever is None:
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
     f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
     Ты - эксперт по ответам на вопросы о содержании документов. Тебе даётся контекст и задаётся вопрос.
     Тебе нужно использовать только данный тебе контекст, чтобы дать максимально подробный ответ на заданный вопрос.
     Важно: никогда не добавляй ничего не из контекста и пиши источники информации (номера и названия пунктов документа). Пиши всегда на русском языке. 
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
    model_inputs = tokenizer([text], return_tensors="pt",truncation=True,max_length=32768).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens = 8192,
        temperature = 0.3,
        #early_stopping = True,
        do_sample = True,
        #num_beams=3
        top_k=50,
        top_p=0.95,
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":
    print(answer_query("О чём написано в пункте 7.1.6?"))