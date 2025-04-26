from fastapi import FastAPI
from pydantic import BaseModel
import requests
import uuid
from transformers import AutoTokenizer
import time

from vtc_multiple import VTCReqQueue
from io_struct import Req
from sampling_params import SamplingParams

app = FastAPI()

@app.get("/status")
def status():
    return {
        "served_tokens_per_user": queue.served,
        "users_in_queue": list(queue.user_req_list.keys()),
        "waiting_reqs": len(queue.waiting_req_list)
    }

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

adapter_dirs = ["alice", "bob", "charlie", "dave"]
fair_weights = [1, 1, 1, 1]

vm_list = [
    "sp25-cs525-0806.cs.illinois.edu",
    "sp25-cs525-0807.cs.illinois.edu",
]

queue = VTCReqQueue(
    max_total_tokens=4096,
    batch_max_tokens=1024,
    running_max_req_size=8,
    adapter_dirs=adapter_dirs,
    fair_weights=fair_weights,
    vm_list=vm_list
)

queue.vm_latency = {vm: 0.5 for vm in vm_list}  # default latency estimates
queue.token_usage = {uid: 0 for uid in adapter_dirs}  # historical token usage

class RequestInput(BaseModel):
    user_id: str
    prompt: str
    max_tokens: int = 100
    stop: list[str] = None

@app.post("/submit")
def submit_request(input: RequestInput):
    request_id = str(uuid.uuid4())
    prompt_ids = tokenizer.encode(input.prompt, add_special_tokens=False)
    sample_params = SamplingParams(
        max_new_tokens=input.max_tokens,
        stop_sequences=[] if input.stop is None else input.stop,
        ignore_eos=False,
        temperature=0.7,
        top_p=1.0,
        top_k=-1
    )
    req = Req(
        adapter_dir=input.user_id,
        request_id=request_id,
        prompt_ids=prompt_ids,
        sample_params=sample_params
    )
    req.submission_time = time.time()
    queue.append(req)

    queue.served[req.adapter_dir] += len(prompt_ids) * queue.input_price / queue.fairw[req.adapter_dir]
    queue.token_usage[req.adapter_dir] += len(prompt_ids)

    return {
        "status": "queued",
        "user": input.user_id,
        "request_id": request_id,
        "prompt_tokens": prompt_ids,
        "prompt_token_count": len(prompt_ids)
    }

@app.post("/dispatch")
def dispatch_to_vllm():
    dispatched = []
    now = time.time()
    all_requests = []

    for req_list in queue.user_req_list.values():
        for req in list(req_list):
            wait_time = now - req.submission_time
            req.current_wait_time = wait_time
            all_requests.append(req)

    if not all_requests:
        return {"dispatched": []}

    max_wait_time = max(req.current_wait_time for req in all_requests)
    max_token_usage = max(queue.token_usage.values()) if queue.token_usage else 1

    alpha = 0.7
    beta = 0.3

    for req in all_requests:
        norm_wait_time = req.current_wait_time / max_wait_time if max_wait_time > 0 else 0
        norm_token_usage = queue.token_usage.get(req.adapter_dir, 0) / max_token_usage
        req.score = alpha * norm_wait_time + beta * (1 - norm_token_usage)

    all_requests.sort(key=lambda r: r.score, reverse=True)

    # Build a to_dispatch list to safely iterate and remove
    to_dispatch = []
    for req in all_requests:
        if req in queue.waiting_req_list:
            to_dispatch.append(req)

    for req in to_dispatch:
        prompt_text = tokenizer.decode(req.prompt_ids)
        payload = {
            "prompt": prompt_text,
            "max_tokens": req.max_output_len,
            "temperature": req.sample_params.temperature,
            "stop": req.sample_params.stop_sequences,
        }

        vm = min(vm_list, key=lambda vm: queue.vm_latency.get(vm, 0))

        try:
            start_time = time.time()
            resp = requests.post(f"http://{vm}:8000/v1/completions", json=payload)
            print(f"📡 Sent to {vm} | Status: {resp.status_code}")
            print(f"📥 Response: {resp.text}")
            end_time = time.time()

            result = resp.json()
            text = result["choices"][0]["text"]
            output_tokens = result["usage"]["completion_tokens"]

            queue.served[req.adapter_dir] += output_tokens * queue.output_price / queue.fairw[req.adapter_dir]
            queue.token_usage[req.adapter_dir] += output_tokens

            dispatched.append((req.request_id, vm, text))

            # ✅ Safe removal
            if req in queue.waiting_req_list:
                queue.waiting_req_list.remove(req)
            else:
                print(f"⚠️ Warning: req {req.request_id} not found in waiting_req_list for user {req.adapter_dir}")

            if req in queue.user_req_list.get(req.adapter_dir, []):
                queue.user_req_list[req.adapter_dir].remove(req)
            else:
                print(f"⚠️ Warning: req {req.request_id} not found in user_req_list[{req.adapter_dir}]")

        except Exception as e:
            dispatched.append((req.request_id, vm, f"Error: {str(e)}"))

    return {"dispatched": dispatched}