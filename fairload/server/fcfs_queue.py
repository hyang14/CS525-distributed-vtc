from collections import deque
from typing import List, Optional

class Req():
    def __init__(self, req_id, adapter_dir, inputs, parameters={}, model_dir="gpt2"):
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.inputs = inputs
        self.parameters = parameters
        self.req_id = req_id
     

class FCFSQueue():
    """
    A class to manage the request queue for a server.
    """
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # self.waiting_req_list: List[Req] = []

        self.user_req_list = deque()
    
    def append(self, req:Req):
        self.user_req_list.append(req)

    # rewrite
    def can_serve(self, req:Req):
        return True

    def generate_next_task(self) -> Optional[Req]:
        if len(self.user_req_list) == 0:
            return None

        next_task = self.user_req_list.popleft()

        return next_task