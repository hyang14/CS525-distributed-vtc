import asyncio
import uuid
from collections import deque
from typing import List, Optional

import numpy as np
import requests

from io_struct import Batch, Req
from infer_utils import calculate_time
from req_queue import ReqQueue


class VTCReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 adapter_dirs, fair_weights, vm_list,
                 input_price=1, output_price=2) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.input_price = input_price
        self.output_price = output_price
        self.served = {}
        self.user_req_list = {}

        self.adapter_dirs = adapter_dirs
        self.fair_weights = fair_weights

        self.vm_list = vm_list
        self.vm_index = 0

        self.fairw = {}
        for i in range(len(adapter_dirs)):
            if i < len(fair_weights):
                self.fairw[adapter_dirs[i]] = fair_weights[i]
            else:
                self.fairw[adapter_dirs[i]] = 1

    # def append(self, req):
    #     self.waiting_req_list.append(req)
    #     if req.adapter_dir not in self.user_req_list:
    #         self.user_req_list[req.adapter_dir] = deque([req])
    #         self.served[req.adapter_dir] = 0
    #     else:
    #         self.user_req_list[req.adapter_dir].append(req)

    #     if len(self.user_req_list[req.adapter_dir]) == 1:
    #         cnts = [v for k, v in self.served.items()
    #                   if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
    #         if len(cnts) > 0:
    #             self.served[req.adapter_dir] = max(self.served[req.adapter_dir], min(cnts))
    def append(self, req):
        self.waiting_req_list.append(req)

        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.served[req.adapter_dir] = 0

            # NEW: assign default fairness weight if missing
            if req.adapter_dir not in self.fairw:
                self.fairw[req.adapter_dir] = 1.0
        else:
            self.user_req_list[req.adapter_dir].append(req)

        if len(self.user_req_list[req.adapter_dir]) == 1:
            cnts = [v for k, v in self.served.items()
                    if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
            if cnts:
                self.served[req.adapter_dir] = max(self.served[req.adapter_dir], min(cnts))


    def _init_cache_list(self, current_batch: Batch, lora_ranks):
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

    def _can_add_new_req(self, req, lora_ranks):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False

    def _get_next_vm(self):
        vm = self.vm_list[self.vm_index]
        self.vm_index = (self.vm_index + 1) % len(self.vm_list)
        return vm

    def generate_parallel_batches(self, current_batches: dict[str, Batch], lora_ranks: dict[str, int]):
        dispatched_batches = []
        for _ in range(len(self.vm_list)):
            if len(self.served) == 0:
                break

            self._init_cache_list(current_batches.get("dummy", None), lora_ranks)
            can_run_list = []
            abort_list = []
            new_batch_total_tokens = 0
            active_served = {k: v for k, v in self.served.items()}

            while True:
                if len(active_served) == 0:
                    break
                adapter_dir = min(active_served, key=active_served.get)
                if len(self.user_req_list[adapter_dir]) > 0:
                    req = self.user_req_list[adapter_dir][0]
                    if req.aborted:
                        abort_list.append(req)
                        self.user_req_list[adapter_dir].popleft()
                        continue
                    if (self._can_add_new_req(req, lora_ranks) and
                        new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                        can_run_list.append(req)
                        new_batch_total_tokens += req.input_len
                        self.user_req_list[adapter_dir].popleft()
                        self.served[adapter_dir] += req.input_len * self.input_price / self.fairw[adapter_dir]
                        active_served[adapter_dir] += req.input_len * self.input_price / self.fairw[adapter_dir]
                    else:
                        break
                else:
                    del active_served[adapter_dir]

            if len(can_run_list) != 0:
                new_batch = Batch(uuid.uuid4().hex, can_run_list)
                self.waiting_req_list = [req for req in self.waiting_req_list
                                         if req not in can_run_list and req not in abort_list]
                vm = self._get_next_vm()
                dispatched_batches.append((new_batch, vm))

        return dispatched_batches

    def update_counter(self, current_batch: Batch):
        for req in current_batch.reqs:
            self.served[req.adapter_dir] += 1 * self.output_price / self.fairw[req.adapter_dir]

    def next_batch(self):
        raise NotImplementedError()