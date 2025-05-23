"""
 Prototype API server for LLM inference request routing and streaming.
 
 Design highlights:
 - FastAPI‑based streaming endpoint `/generate_stream`.
 - Pluggable LoadBalancer, Scheduler, and EngineClient components.
 - Non‑blocking, back‑pressure‑aware streaming to clients.
 - Easy to extend: implement new load balancer strategies, schedulers, or engine adapters.
 """

# 512 token max per vm

import argparse
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, AsyncGenerator, Optional
import asyncio
import httpx
import json
import uuid
import logging
import aiohttp

import uvicorn

from .req_queue import ReqQueue, Req
from .fcfs_queue import FCFSQueue
from .lat_queue import LatQueue

# ---------- Config -----------------------------------------------------------------

max_total_tokens = 512
batch_max_tokens = 512
running_max_req_size = 512

isFirst = True

model = "Qwen/Qwen2.5-0.5B"

TIMEOUT_KEEP_ALIVE = 5
ENGINE_ENDPOINTS = [
    "http://localhost:8000/v1",  # engine A
    # "http://localhost:8000/v1",  # engine B
]  # Add more endpoints any time

# ---------- FastAPI Setup ----------------------------------------------------------

app = FastAPI()

scheduler = ReqQueue(max_total_tokens, batch_max_tokens, running_max_req_size)
# scheduler = FCFSQueue(max_total_tokens, batch_max_tokens, running_max_req_size)
# scheduler = LatQueue(max_total_tokens, batch_max_tokens, running_max_req_size)
request_queues: Dict[str, asyncio.Queue[bytes]] = {} 
pending_event   = asyncio.Event()  

@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"

async def scheduler_loop() -> None:
    async with httpx.AsyncClient() as session:
        while True:

            # print("after new request received")
            # print([(user, len(requests)) for user, requests in scheduler.user_req_list.items()])
            # pull the next request according to your ReqQueue policy
            req = scheduler.generate_next_task()        # **your fairness logic**
            if req is None:
                # print("No request to process")
                # wait until work is available
                await pending_event.wait()
                pending_event.clear()
                
                continue
            # print("after new request scheduled")
            print(req)
            # print([(user, len(requests)) for user, requests in scheduler.user_req_list.items()])

            # Pull out the queue we’ll write chunks into
            q = request_queues.get(req.req_id)
            if q is None:       # client vanished before we started
                print(f"Client vanished for req_id {req.req_id}")
                continue

            # actually call the engine and stream the bytes
            payload = {
                "model": model,
                "prompt": req.inputs,
                "stream": True,
                # **req.parameters,
            }
            try:
                engine_url = f"{ENGINE_ENDPOINTS[0]}/completions"
                async with session.stream("POST", engine_url,
                                           json=payload, timeout=None) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"engine returned {resp.status_code}")
                    # print(resp)
                    async for chunk in resp.aiter_text():
                        if not chunk.startswith("data: "):
                            continue
                        finished = False
                        try:
                            json_str = chunk[len("data: "):]
                            if json_str.strip() == "[DONE]":
                                finished = True
                                break
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue  # skip malformed chunks
                        ret = {
                            "token": {
                                "id": req.req_id,
                                "text": data["choices"][0].get("text", ""),
                                "logprob": None,
                                "special": False
                            },
                            "generated_text": None,
                            "finished": finished,
                            "details": None
                        }

                        await q.put(("data:" + json.dumps(ret, ensure_ascii=False) + f"\n\n").encode("utf-8"))
                        scheduler.update_token(req)      # keep fairness counters
                        

            except Exception as e:
                await q.put(f"data: {json.dumps({'error': str(e)})}\n\n".encode())

            # tell the client we’re done
            # scheduler.update_time(req)
            await q.put(None)

async def generate(prompt: str):
    session = httpx.AsyncClient()
    metadata = {}
    finished = False
    url = f"{ENGINE_ENDPOINTS[0]}/completions"

    payload = {
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt": prompt,
        "stream": True
    }

    async with session.stream("POST", url, json=payload, timeout=None) as resp:
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Engine error {resp.status_code}")
        async for chunk in resp.aiter_text():
            # print(chunk)
            if chunk.startswith("data: "):
                try:
                    json_str = chunk[len("data: "):]
                    if json_str.strip() == "[DONE]":
                        finished = True
                        break
                    data = json.loads(json_str)
                    # print(data)
                except json.JSONDecodeError:
                    continue  # skip malformed chunks
                
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": data["choices"][0].get("text", ""),
                    "logprob": metadata.get("logprob", None),
                    "special": False
                },
                "generated_text": None,
                "finished": finished,
                "details": None
            }
            yield ("data:" + json.dumps(ret, ensure_ascii=False) + f"\n\n").encode(
                "utf-8"
            )

@app.post("/generate_stream")
async def generate_stream(request: Request):
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(scheduler_loop())
        isFirst = False

    data = await request.json()
    prompt    = data.get("inputs", "")
    adapter   = data.get("lora_dir", "default")
    params    = data.get("parameters", {})
    model_dir = data.get("model_dir", "gpt2")
    req_id    = data.get("req_id", str(uuid.uuid4()))

    req = Req(req_id, adapter, prompt, params, model_dir)
    scheduler.append(req)
    q = asyncio.Queue[bytes]()
    request_queues[req_id] = q

    pending_event.set()

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            while True:
                chunk = await q.get()               # blocks until something written
                if chunk is None:                   # sentinel → finished
                    break
                yield chunk
        finally:                                    # client disconnected or finished
            request_queues.pop(req_id, None)
    # print('error check')
    return StreamingResponse(event_stream(), media_type="text/event-stream")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)

    args = parser.parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )

if __name__ == "__main__":
    main()
