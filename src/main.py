import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever import retrieve
from db_creation import create_db

import config

def answer_query(query, 
                 model_name=config.DEFAULT_MODEL,
                 embeddings_name=config.DEFAULT_EMBEDDINGS,
                 persistent_directory = config.DEFAULT_DB_DIR,
                 doc_path = config.DEFAULT_DOC_FILE,
                 model = None,
                 tokenizer = None,
                 embeddings = None,
                 db = None,
                 text_splitter = None,
                 relevant_docs = [],
                 new_db = False,
                 show_docs = False):
    
    if relevant_docs == []:
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)

        if not os.path.exists(persistent_directory) or new_db:
            print("Создание БД")
            db = create_db(doc_path, persistent_directory, embeddings, text_splitter)
        else:
            print("БД уже существует")
            db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)            

        relevant_docs = retrieve(query, db, ret_trust=0.4, k_doc_orig=40, k_doc_final=15)

        if show_docs:
            for i, doc in enumerate(relevant_docs, 1):
                print(f"Документ {i}:\n{doc}\n")

    context_docs = "\n\n".join([doc for doc in relevant_docs])

    if model is None: model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    if tokenizer is None: tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
    {"role": "system", "content": 
     f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
     Ты - эксперт по ответам на вопросы о содержании документов. Тебе даётся контекст и задаётся вопрос.
     Тебе нужно использовать только данный тебе контекст, чтобы дать максимально подробный ответ на заданный вопрос.
     Важно: никогда не добавляй ничего не из контекста и пиши источники информации (номера и названия пунктов документа). Пиши всегда на русском языке. Не делай ссылки на контекст.
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
        max_new_tokens=8192,
        temperature=0.4,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":
    print(answer_query("Опиши менеджмент человеческих ресурсов проекта", show_docs=True))