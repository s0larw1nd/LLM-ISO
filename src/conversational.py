import os
from main import answer_query
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def format_chat_history(chat_history):
    """
    return "\n".join(
        f"Пользователь: {msg.content}" if isinstance(msg, HumanMessage) 
        else f"AI: {msg.content}" 
        for msg in chat_history
    )    
    """
    return "\n".join(
        f"{k}: {msg[k]}"
        for msg in chat_history
        for k in msg
    )

def start_conversation(model_name="./models/Qwen2.5-7B-Instruct-merged",
                       embeddings_name="./models/FRIDA",
                       persistent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db"),
                       file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "doc.docx")):
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)

    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15},
    )

    chat_history = []

    while True:
        query = input("Вопрос: ")

        if query.lower() == "выход": break

        messages = [
        {"role": "system", "content": f"""
        Используя историю диалога, тебе нужно переписать последний вопрос пользователя, который может ссылаться на информацию в истории, 
        в полностью самостоятельный вопрос, на который можно ответить без знания истории диалога. Важно: не отвечай на вопрос, только переформулируй.
        Если вопрос не нуждается в подобной переформулировке, верни изначальный вариант. При формулировании старайся внести минимальные изменения, которые позволят добиться цели.
        Старайся избегать сильного изменения вопроса, если того не требуется.
        Всегда отвечай в формате одного вопроса на русском языке.
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
            max_new_tokens=8192
        )

        new_query = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, new_query)]

        new_query = tokenizer.batch_decode(new_query, skip_special_tokens=True)[0]

        relevant_docs = retriever.invoke(new_query)

        answer = answer_query(query=new_query, model=model, tokenizer=tokenizer, context_docs=relevant_docs)

        print("Ответ:")
        print(answer)
        print()

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))

def answer_history(chat_history,
                   model,
                   tokenizer,
                   retriever,
                   embeddings):

    query = chat_history[-1]["user"]
    
    messages = [
        {"role": "system", "content": f"""
        Используя историю диалога, тебе нужно переписать последний вопрос пользователя, который может ссылаться на информацию в истории, 
        в полностью самостоятельный вопрос, на который можно ответить без знания истории диалога. Важно: не отвечай на вопрос, только переформулируй.
        Если вопрос не нуждается в подобной переформулировке, верни изначальный вариант. При формулировании старайся внести минимальные изменения, которые позволят добиться цели.
        Старайся избегать сильного изменения вопроса, если того не требуется.
        Всегда отвечай в формате одного вопроса на русском языке.
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
        max_new_tokens=8192
    )

    new_query = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, new_query)]

    new_query = tokenizer.batch_decode(new_query, skip_special_tokens=True)[0]

    relevant_docs = retriever.invoke(new_query)

    answer = answer_query(query=new_query, model=model, tokenizer=tokenizer, context_docs=relevant_docs, embeddings=embeddings)

    return answer

if __name__ == "__main__":
    start_conversation()