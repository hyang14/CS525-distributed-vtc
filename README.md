# CS525-distributed-vtc

Dependency Install

`pip install -r requirements.txt`

Experiment Sequence

- launch scheduler `python FairLoad/launch_server.py`  use `python -m fairload.server.api_server` for now
- launch serving engines
- lauch experiment `python experiment/run_exp.py`

test code 
`python experiment/run_exp.py --debug`