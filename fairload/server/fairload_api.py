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

from req_queue import ReqQueue

# ---------- Config -----------------------------------------------------------------

max_total_tokens = 512
batch_max_tokens = 512
running_max_req_size = 512

TIMEOUT_KEEP_ALIVE = 5

ENGINE_ENDPOINTS = [
    "http://localhost:8000/v1",  # engine A
    # "http://localhost:8000/v1",  # engine B
]  # Add more endpoints any time

# ---------- Models -----------------------------------------------------------------


class GenerationParameters(BaseModel):
    do_sample: bool = False
    ignore_eos: bool = True
    max_new_tokens: int = 128
    temperature: Optional[float] = None


class GenerateRequest(BaseModel):
    model_dir: str
    lora_dir: Optional[str] = Field(None, alias="adapter_dir")
    inputs: str = Field(..., alias="prompt")
    parameters: GenerationParameters
    req_id: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


# ---------- Core Components ---------------------------------------------------------


class EngineClient:
    """Adapter responsible for talking to an OpenAI‑compatible engine and yielding
    partial tokens as they arrive via SSE/streaming HTTP."""

    def __init__(self, engine_base_url: str):
        self.engine_base_url = engine_base_url.rstrip("/")
        self.session = httpx.AsyncClient()

    async def stream_generate(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        url = f"{self.engine_base_url}/v1/completions"
        async with self.session.stream("POST", url, json=payload, timeout=None) as resp:
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Engine error {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data_part = line[len("data:") :]
                    if data_part.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data_part)
                    except json.JSONDecodeError:
                        logging.warning("Malformed JSON from engine: %s", data_part)
                else:
                    logging.debug("Skipping non‑data line: %s", line)

    async def close(self):
        await self.session.aclose()


class RoundRobinLoadBalancer:
    """Very simple round‑robin load balancer across a fixed set of engine endpoints."""

    def __init__(self, engine_endpoints: List[str]):
        self.engines = engine_endpoints
        self._idx = 0
        self._lock = asyncio.Lock()

    async def pick_engine(self) -> str:
        async with self._lock:
            url = self.engines[self._idx]
            self._idx = (self._idx + 1) % len(self.engines)
            return url


class RequestContext:
    """Wraps everything the scheduler needs to process a request."""

    def __init__(self, data: Dict[str, Any], out_q: "asyncio.Queue[Optional[bytes]]"):
        self.data = data
        self.req_id = data["req_id"]
        self.out_q = out_q
        # tracking fields (tokens used, priority, deadlines…) can be added later


class FIFOScheduler:
    """Simple FIFO scheduler: drain incoming requests queue and dispatch to engines.
    Replace/extend for more sophisticated fairness or QoS."""

    def __init__(self, lb: RoundRobinLoadBalancer):
        self._in_q: "asyncio.Queue[RequestContext]" = asyncio.Queue()
        self._lb = lb
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    async def submit(self, ctx: RequestContext):
        await self._in_q.put(ctx)

    async def abort(self, req_id: str):
        task = self._tasks.pop(req_id, None)
        if task:
            task.cancel()

    async def run(self):
        while not self._shutdown_event.is_set():
            ctx: RequestContext = await self._in_q.get()
            engine_url = await self._lb.pick_engine()
            logging.debug("Dispatching %s to %s", ctx.req_id, engine_url)
            task = asyncio.create_task(self._handle_request(ctx, engine_url))
            self._tasks[ctx.req_id] = task

    async def _handle_request(self, ctx: RequestContext, engine_url: str):
        engine = EngineClient(engine_url)
        try:
            async for payload in engine.stream_generate(ctx.data):
                token_text = payload.get("token", {}).get("text", payload.get("text", ""))
                finished = payload.get("finished", False)
                ret = {
                    "token": {
                        "id": payload.get("token", {}).get("id", None),
                        "text": token_text,
                        "logprob": payload.get("token", {}).get("logprob", None),
                        "special": False,
                    },
                    "generated_text": None,
                    "finished": finished,
                    "details": None,
                }
                await ctx.out_q.put(("data:" + json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf‑8"))
                if finished:
                    break
        except Exception as exc:
            logging.exception("Engine stream failed for %s", ctx.req_id)
            await ctx.out_q.put(("data:" + json.dumps({"details": str(exc)}, ensure_ascii=False) + "\n\n").encode("utf‑8"))
        finally:
            await ctx.out_q.put(None)  # signal end of stream
            await engine.close()

    async def shutdown(self):
        self._shutdown_event.set()
        for t in self._tasks.values():
            t.cancel()


# ---------- FastAPI Setup ----------------------------------------------------------

app = FastAPI()

load_balancer = RoundRobinLoadBalancer(ENGINE_ENDPOINTS)
# scheduler = FIFOScheduler(load_balancer)
scheduler = ReqQueue(max_total_tokens, batch_max_tokens, running_max_req_size)

@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"

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
    data = await request.json()
    prompt = data.get("prompt", "Hello world")
    return StreamingResponse(generate(prompt), media_type="text/plain")

# async def generate_stream(request: Request, background_tasks=None):
#     req_id = request.req_id or str(uuid.uuid4())
#     request_dict = await request.json()
#     # request_payload = request.dict(by_alias=True)
#     adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
#     prompt = request_dict.pop("inputs")
#     request_dict["req_id"] = req_id

    # out_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=100)

    # ctx = RequestContext(request_payload, out_q)
    # await scheduler.submit(ctx)

    # async def stream_results() -> AsyncGenerator[bytes, None]:
    #     while True:
    #         chunk = await out_q.get()
    #         if chunk is None:
    #             break
    #         yield chunk

    # async def abort_request() -> None:
    #     await scheduler.abort(req_id)
    
    # if background_tasks is None:
    #     background_tasks = BackgroundTasks()
    # background_tasks.add_task(abort_request)

    # return StreamingResponse(
    #     stream_results(),
    #     media_type="text/event-stream",
    #     headers={"X-Request-ID": req_id},
    #     background=background_tasks,
    # )

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
