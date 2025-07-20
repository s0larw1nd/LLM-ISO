from fastapi import FastAPI
from pydantic import BaseModel

from conversational import answer_history
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

from typing import Any

import config

model_name = config.DEFAULT_MODEL
embeddings_name = config.DEFAULT_EMBEDDINGS
persistent_directory = config.DEFAULT_DB_DIR

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map="auto", 
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
db = Chroma(
    persist_directory=persistent_directory, 
    embedding_function=embeddings
    )

app = FastAPI()

class ChatRequest(BaseModel):
    history: Any

@app.post("/answer")
async def answer(request: ChatRequest):
    print(request.history)
    response = answer_history(request.history, model=model, tokenizer=tokenizer, db=db, embeddings=embeddings)

    return {"response": response}