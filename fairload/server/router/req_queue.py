import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
# from slora.utils.infer_utils import  calculate_time


class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        # Debug and adapt to new sering engine
        adapter_rank = lora_ranks.get(req.adapter_dir, 0)
        self.adapter_size += adapter_rank * 4
        self.adapters.add(req.adapter_dir)

        # if req.adapter_dir not in self.adapters:
        #     self.adapter_size += lora_ranks[req.adapter_dir] * 4
        #     self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False
    
    def update_counter(self, req):
        pass 

    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None


    def next_batch(self):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None


    def cost_func_profile(self, num_input_token, num_output_token):
        x, y = num_input_token, num_output_token
        return (0.21001891 * x + 0.10075395 * y +
                0.0039868 * x * y + 0.00324733 * y * y + 1.14601454)

    # additional function for scheduler 
    def is_empty(self) -> bool:
        return len(self.waiting_req_list) == 0

    def finish_req(self, req: Req) -> None:
        # remove it from the accounting structures the scheduler keeps
        if req in self.running_req_list:
            self.running_req_list.remove(req)
        # Many schedulers track per-user deficit counters; update them.
        self.update_after_finish(req)      # exists in VTC/* queues
