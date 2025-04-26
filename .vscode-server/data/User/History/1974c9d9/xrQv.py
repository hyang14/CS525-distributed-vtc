class VTCReqQueue(ReqQueue):
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 vm_list: list[str],
                 input_price=1, output_price=2) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.input_price = input_price
        self.output_price = output_price

        self.served = {}          # user_id -> virtual tokens spent
        self.user_req_list = {}   # user_id -> deque of requests

        self.vm_list = vm_list
        self.vm_index = 0

    def append(self, req):
        self.waiting_req_list.append(req)
        uid = req.user_id
        if uid not in self.user_req_list:
            self.user_req_list[uid] = deque([req])
            self.served[uid] = 0
        else:
            self.user_req_list[uid].append(req)

        if len(self.user_req_list[uid]) == 1:
            cnts = [v for k, v in self.served.items() if len(self.user_req_list[k]) > 0 and k != uid]
            if cnts:
                self.served[uid] = max(self.served[uid], min(cnts))

    def _can_add_new_req(self, req, current_total_tokens, current_req_count):
        est_tokens = current_total_tokens + req.input_len
        return est_tokens <= self.max_total_tokens and current_req_count < self.running_max_req_size

    def _get_next_vm(self) -> str:
        vm = self.vm_list[self.vm_index]
        self.vm_index = (self.vm_index + 1) % len(self.vm_list)
        return vm

    def generate_parallel_batches(self, current_batches: dict[str, Batch] = None) -> List[tuple[Batch, str]]:
        """
        Generate up to N batches for N different VMs and return them as (batch, vm) tuples.
        """
        if current_batches is None:
            current_batches = {}

        batches_to_dispatch = []
        local_served = dict(self.served)  # Don't mutate real state until dispatch confirmed

        for _ in range(len(self.vm_list)):
            vm = self._get_next_vm()

            if vm in current_batches and len(current_batches[vm].reqs) >= self.running_max_req_size:
                continue

            can_run_list = []
            new_batch_total_tokens = 0
            active_served = dict(local_served)
            abort_list = []

            while active_served:
                uid = min(active_served, key=active_served.get)
                if len(self.user_req_list[uid]) == 0:
                    del active_served[uid]
                    continue

                req = self.user_req_list[uid][0]
                if req.aborted:
                    abort_list.append(req)
                    self.user_req_list[uid].popleft()
                    continue

                if self._can_add_new_req(req, new_batch_total_tokens, len(can_run_list)):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    self.user_req_list[uid].popleft()

                    cost = req.input_len * self.input_price
                    local_served[uid] += cost
                    active_served[uid] += cost
                else:
                    break

            if can_run_list:
                self.waiting_req_list = [r for r in self.waiting_req_list if r not in can_run_list and r not in abort_list]
                batch = Batch(uuid.uuid4().hex, can_run_list)
                batches_to_dispatch.append((batch, vm))

                # commit updates only if batch is successful
                for req in can_run_list:
                    self.served[req.user_id] = local_served[req.user_id]

        return batches_to_dispatch

    def update_counter(self, current_batch: Batch):
        for req in current_batch.reqs:
            uid = req.user_id
            self.served[uid] += self.output_price
