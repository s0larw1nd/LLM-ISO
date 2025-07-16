import asyncio
import json
import uuid
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import httpx
import redis

request_queue = asyncio.Queue()

async def worker():
    async with httpx.AsyncClient() as client:
        while True:
            history, future = await request_queue.get()
            print(f"Обработка {history}")
            try:
                response = await client.post(
                    "http://localhost:8001/answer",
                    json={"history": history},
                    timeout=100
                )
                result = response.json()
                future.set_result({"result": result})
            except Exception as e:
                future.set_exception(e)
            finally:
                request_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(worker())
    yield

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
    history.append({"assistant": response['response']})
    redis_client.setex(f"chat:{request.session_id}", 600, json.dumps(history))

    return {"response": response}