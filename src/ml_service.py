from fastapi import FastAPI
from pydantic import BaseModel

from conversational import answer_history
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

from typing import Any

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

app = FastAPI()

class ChatRequest(BaseModel):
    history: Any

@app.post("/answer")
async def answer(request: ChatRequest):
    print(request.history)
    response = answer_history(request.history, model=model, tokenizer=tokenizer, retriever=retriever, embeddings=embeddings)

    return {"response": response}