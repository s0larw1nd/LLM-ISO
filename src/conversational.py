import os
from main import answer_query
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from retriever import retrieve

import config

def format_chat_history(chat_history):
    return "\n".join(
        f"{k}: {msg[k]}"
        for msg in chat_history
        for k in msg
    )

def answer_history(chat_history,
                   model,
                   tokenizer,
                   db,
                   embeddings):

    query = chat_history[-1]["user"]
    
    messages = [
        {"role": "system", "content": f"""
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
        Ты - эксперт по переписыванию вопросов. Тебе дана история диалога и последний вопрос пользователя. Определи, опирается ли последний вопрос пользователя на информацию из истории диалога.
        Если да, то перепиши его таким образом, чтобы он включал в себя весь нужный контекст. Иначе - оставь в изначальном состоянии.
        Важно: не отвечай на вопрос и не добавляй ничего не из истории диалога, только перепиши, сохранив смысл.
        Отвечай в формате одного вопроса на русском языке.
        ИСТОРИЯ:
        {format_chat_history(chat_history)}
        КОНЕЦ ИСТОРИИ"""
        },
        {"role": "user", "content": "ВОПРОС: " + query}
    ]
    
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([text], return_tensors="pt",truncation=True,max_length=32768).to(model.device)

    new_query = model.generate(
        **model_inputs,
        max_new_tokens=8192,
        temperature=0.5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    new_query = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, new_query)]

    new_query = tokenizer.batch_decode(new_query, skip_special_tokens=True)[0]

    relevant_docs = retrieve(new_query, db)

    answer = answer_query(query=new_query, model=model, tokenizer=tokenizer, relevant_docs=relevant_docs, embeddings=embeddings)

    return answer

def start_conversation(model_name=config.DEFAULT_MODEL,
                       embeddings_name=config.DEFAULT_EMBEDDINGS,
                       persistent_directory=config.DEFAULT_DB_DIR):
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)

    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    chat_history = []

    while True:
        query = input("Вопрос: ")

        if query.lower() == "выход": break

        chat_history.append({"user": query})

        answer = answer_history(chat_history, model, tokenizer, db, embeddings)

        print("Ответ:")
        print(answer)
        print()

        chat_history.append({"assistant": answer})

if __name__ == "__main__":
    start_conversation()