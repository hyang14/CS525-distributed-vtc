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
    req.submission_time = time.time()  # record submission time
    queue.append(req)

    # ✅ Update input token accounting (w_p) immediately upon queuing
    queue.served[req.adapter_dir] += len(prompt_ids) * queue.input_price / queue.fairw[req.adapter_dir]

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
    current_batches = {"dummy": None}
    lora_ranks = {adapter: 1 for adapter in queue.user_req_list.keys()}

    output_token_counts = {}
    input_token_counts = {}
    token_summary_per_request = {}
    response_times = {}
    latencies = {}
    for batch, vm in queue.generate_parallel_batches(current_batches, lora_ranks):
        print(f"🚀 Dispatching batch {batch.batch_id} to {vm}")

        for req in batch.reqs:
            try:
                prompt_text = tokenizer.decode(req.prompt_ids)
            except Exception as e:
                prompt_text = "[Decode Error]"

            payload = {
                "prompt": prompt_text,
                "max_tokens": req.max_output_len,
                "temperature": req.sample_params.temperature,
                "stop": req.sample_params.stop_sequences,
            }

            try:
                start_time = time.time()
                resp = requests.post(f"http://{vm}:8000/v1/completions", json=payload)
                end_time = time.time()
                json_resp = resp.json()
                result = json_resp["choices"][0]["text"]
                actual_output_tokens = json_resp["usage"]["completion_tokens"]
                dispatched.append((vm, result))
                response_times.setdefault(req.adapter_dir, []).append(end_time - start_time)
                if hasattr(req, 'submission_time'):
                    latencies.setdefault(req.adapter_dir, []).append(end_time - req.submission_time)

                output_token_counts.setdefault(req.adapter_dir, 0)
                input_token_counts.setdefault(req.adapter_dir, 0)
                token_summary_per_request[req.request_id] = {
                    "input_tokens": len(req.prompt_ids),
                    "output_tokens": actual_output_tokens
                }
                output_token_counts[req.adapter_dir] += actual_output_tokens
                input_token_counts[req.adapter_dir] += len(req.prompt_ids)
                queue.served[req.adapter_dir] += actual_output_tokens * queue.output_price / queue.fairw[req.adapter_dir]

            except Exception as e:
                dispatched.append((vm, f"Error: {str(e)}"))

    return {
        "dispatched": dispatched,
        "output_token_counts": output_token_counts,
        "input_token_counts": input_token_counts,
        "token_summary_per_request": token_summary_per_request,
        "response_times": response_times,
        "latencies": latencies
    }




