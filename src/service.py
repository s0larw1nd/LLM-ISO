import asyncio
from concurrent.futures import ProcessPoolExecutor
import json
import uuid
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel

from conversational import answer_history
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import redis

request_queue = asyncio.Queue()
process_pool = ProcessPoolExecutor(max_workers=1)

async def worker():
    model_name = "./models/Qwen2.5-7B-Instruct-merged"
    embeddings_name="./models/FRIDA"
    persistent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto", 
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    retriever = Chroma(
        persist_directory=persistent_directory, 
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 15})

    while True:
        history, future = await request_queue.get()

        print(f"Обработка {history}")

        try:
            result = answer_history(history, model=model, tokenizer=tokenizer, retriever=retriever, embeddings=embeddings)
            future.set_result({"result": result})
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(worker())
    yield
    process_pool.shutdown()

app = FastAPI(lifespan=lifespan)
redis_client = redis.Redis(host="localhost", port=8008, db=0)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/start")
async def start_chat():
    session_id = str(uuid.uuid4())
    redis_client.setex(f"chat:{session_id}", 600, json.dumps([]))
    return {"session_id": session_id}

@app.post("/chat")
async def chat(request: ChatRequest):
    history = json.loads(redis_client.get(f"chat:{request.session_id}") or "[]")
    history.append({"user": request.message})
    future = asyncio.Future()

    await request_queue.put((history, future))
    
    result_dict = await future
    response = result_dict["result"]
    history.append({"assistant": response})
    redis_client.setex(f"chat:{request.session_id}", 600, json.dumps(history))

    return {"response": response, "history": history}