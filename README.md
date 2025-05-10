# CS525-distributed-vtc

Dependency Install

`pip install -r requirements.txt`

Experiment Sequence

- launch scheduler `python FairLoad/launch_server.py`  use `python -m fairload.server.api_server` for now
- launch serving engines
- lauch experiment `python experiment/run_exp.py` use `python experiment/run_exp.py --suite overload --output result/debug.jsonl` for now

test code 
`python experiment/run_exp.py --debug`


0808, 0809, 0811,0812,0813 all have vllm installed


0806 as main dispatching engine


0806, 0807, 0808, 0809, 0810 have dependencies for load balancer

To start vllm on serving engines, 
```
sudo docker run --rm \
  --shm-size=2g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.huggingface:/root/.huggingface \
  -p 8000:8000 \
  public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.8.5 \
  --model facebook/opt-125m \
  --swap-space 2 \
  --tensor-parallel-size 1 \
  --disable-async-output-proc
```
If there is no output from the vllm side when sending request, checked 0806's terminal window where it runned the fairload.server.fairload_api and see if there is an error related to pending event starting. 
If so, change pending event initialization to
```
pending_event   = None 
@app.on_event("startup")
async def startup_event():
    global pending_event
    pending_event = asyncio.Event()
```
Or see minor_fix_fcfs branch for more details.

test