from collections import deque
import time
from typing import List, Optional

class Req():
    def __init__(self, req_id, adapter_dir, inputs, parameters={}, model_dir="gpt2"):
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.inputs = inputs
        self.parameters = parameters
        self.req_id = req_id
     

class LatQueue():
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
        self.last_token_time = {}
        self.active_user_req = {}

    # update time when request is done
    def update_time(self, req:Req):
        self.last_token_time[req.adapter_dir] = time.time()
        self.active_user_req[req.adapter_dir] = max(0, self.active_user_req[req.adapter_dir] - 1)

        # if req.adapter_dir not in self.token_served:
        #     self.token_served[req.adapter_dir] = 0
        # self.token_served[req.adapter_dir] += 1
    
    def append(self, req:Req):
        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
        else:
            self.user_req_list[req.adapter_dir].append(req)

        # waiting queue was empty before
        if len(self.user_req_list[req.adapter_dir]) == 1:
            self.last_token_time[req.adapter_dir] = time.time()
            self.last_token_time[req.adapter_dir] = max(self.last_token_time[req.adapter_dir], min([v for v in self.last_token_time.values()])) # there is active req
            # self.active_user_req[req.adapter_dir] = 1

            # # lift counter
            # time = [v for k, v in self.last_token_time.items()
            #           if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
            # if len(time) > 0:
            #     self.token_served[req.adapter_dir] = max(self.token_served[req.adapter_dir], min(cnts))
            # else:
            #     self.token_served[req.adapter_dir] = 0

    # rewrite
    def can_serve(self, req:Req):
        return True

    # generate next task based time since last token generated, if the smallest is inf, generate next task based on fewest active  user req
    def generate_next_task(self) -> Optional[Req]:
        if not self.user_req_list:               # empty
            return None
        
        waiting_clients = {client_id for client_id, reqs in self.user_req_list.items() if len(reqs) > 0}
        if len(waiting_clients) == 0:
            return None
        
        # filter out client with waiting reqs
        next_clients = [(client_id, self.last_token_time[client_id]) for client_id in waiting_clients if self.last_token_time[client_id] < float('inf')]

        # if all clients have no waiting reqs, use active user req
        if len(next_clients) == 0:  
            next_clients = [(client_id, self.active_user_req[client_id]) for client_id in waiting_clients if self.active_user_req[client_id] > 0]
    
        client = min(next_clients, key=lambda x: x[1])[0]
        next_task = self.user_req_list[client].popleft()
        if client not in self.active_user_req:
            self.active_user_req[client] = 1
        else:
            self.active_user_req[client] += 1
        self.last_token_time[client] = float('inf') # there is active req

        return next_task
    



        


        # active_served = {k: v for k, v in self.token_served.items() if len(self.user_req_list[k]) > 0}
        # if not active_served:                    # all empty
        #     return None
        # adapter_dir = min(active_served, key=active_served.get)

        # if not self.user_req_list[adapter_dir]:
        #     return None

        # req = self.user_req_list[adapter_dir][0]

        # if not self.can_serve(req):
        #     return None
        # next_task = self.user_req_list[adapter_dir].popleft()

        # return next_task