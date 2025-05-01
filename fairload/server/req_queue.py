from collections import deque
from typing import List

class Req():
    def __init__(self, req_id, adapter_dir, inputs, parameters={}, model_dir="gpt2"):
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.inputs = inputs
        self.parameters = parameters
        self.req_id = req_id
     

class ReqQueue():
    """
    A class to manage the request queue for a server.
    """
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # self.waiting_req_list: List[Req] = []

        self.user_req_list = {}
        self.token_severed = {}

    def update_token(self, req:Req):
        if req.adapter_dir not in self.token_severed:
            self.token_severed[req.adapter_dir] = 0
        self.token_severed[req.adapter_dir] += 1
    
    def append(self, req:Req):
        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
        else:
            self.user_req_list[req.adapter_dir].append(req)

        # waiting queue was empty before
        if len(self.user_req_list[req.adapter_dir]) == 1:
            # lift counter
            cnts = [v for k, v in self.token_severed.items()
                      if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
            if len(cnts) > 0:
                self.token_severed[req.adapter_dir] = max(self.token_severed[req.adapter_dir], min(cnts))

    # rewrite
    def can_serve(self, req:Req):
        return True

    def generate_next_task(self):
        active_served = {k: v for k, v in self.served.items()}
        adapter_dir = min(active_served, key=active_served.get)
        req = self.user_req_list[adapter_dir][0]
        
        # wait until a server is available
        while True:
            if self.can_serve(req):
                break
        return self.user_req_list[adapter_dir].popleft()


