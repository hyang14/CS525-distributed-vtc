import zmq
import zmq.asyncio
import asyncio
import uvloop
from typing import Union

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq, BatchAbortReq


class HttpServerManager:
    def __init__(
        self,
        model_weightdir,
        tokenizor_mode,
        router_port,
        httpserver_port,
        total_token_num,
        max_req_input_len,
        max_req_total_len,
        trust_remote_code,
        dummy=False,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        #debug
        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")
        # self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")

        try: 
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code) 
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode) 

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.total_token_num = total_token_num
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len

    async def generate(self, adapter_dir, prompt, sampling_params, request_id):

        # prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = [prompt]
        prompt_tokens = len(prompt_ids)
        if prompt_tokens > self.max_req_input_len:
            raise ValueError(
                f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}"
            )
        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        # DEBUG Directly yield the prompt as output
        # metadata = {"prompt_tokens": prompt_tokens}
        # finished = True
        # yield prompt, metadata, finished

        # sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        # send the request to the router
        self.send_to_router.send_pyobj((adapter_dir, prompt_ids, sampling_params, request_id))
        #display the request that router will receive

        event = asyncio.Event()
        self.req_id_to_out_inf[request_id] = ("", {}, False, event)
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            event.clear()
            # request_id is aborted by the backend system for traffic control
            if request_id not in self.req_id_to_out_inf:
                yield "", {}, -1
                break
            out_str, metadata, finished, _ = self.req_id_to_out_inf[request_id]
            if len(metadata) != 0:
                self.req_id_to_out_inf[request_id] = ("", {}, finished, event)
                metadata["prompt_tokens"] = prompt_tokens
                yield out_str, metadata, finished
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                break
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        try:
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans:Union(BatchStrOut, BatchAbortReq) = await self.recv_from_detokenization.recv_pyobj()
            print(recv_ans)
            
            #debug
            # if isinstance(recv_ans, tuple) and len(recv_ans) == 4:
            #     adapter_dir, prompt_ids, sampling_params, request_id = recv_ans
            #     # print(f"recv from detokenization: {adapter_dir}, {prompt_ids}, {sampling_params}, {request_id}")
                
            #     # Simple simulation: just return the decoded prompt as response
            #     prompt_text = prompt_ids
                
            #     # Simulate token generation (just returning the prompt for now)
            #     # You could implement more complex token generation logic here
            #     response_text = prompt_text.append(" [SIMULATED RESPONSE]")
                
            #     # Create metadata with token counts
            #     metadata = {
            #         "generated_tokens": len(prompt_ids),
            #         "simulation": True
            #     }
                
            #     # Create a BatchStrOut response
            #     batch_out = BatchStrOut()
            #     batch_out.reqs_infs = [(request_id, response_text, metadata, True, False)]
            #     recv_ans = batch_out
            # else:
            #     # Unknown message type, create an empty response
            #     print("unknown message type, create an empty response")
            #     recv_ans = BatchStrOut()
            #     recv_ans.reqs_infs = []
            # #test
            # adapter_dir, prompt_ids, sampling_params, request_id = recv_ans
            # temp = BatchStrOut()
            # temp.reqs_infs = [(request_id,prompt_ids,{},True,False)]
            # recv_ans = temp
            # assert isinstance(recv_ans, (BatchStrOut, BatchAbortReq)), f"error recv type {type(recv_ans)}"
            if isinstance(recv_ans, BatchStrOut):
                for req_id, text, metadata, finished, abort in recv_ans.reqs_infs:
                    try:
                        if not abort:
                            _, _, _, event = self.req_id_to_out_inf[req_id]
                            self.req_id_to_out_inf[req_id] = (
                                text,
                                metadata,
                                finished,
                                event,
                            )
                            event.set()
                        else:
                            del self.req_id_to_out_inf[req_id]
                    except:
                        pass
            elif isinstance(recv_ans, BatchAbortReq):
                print("abort reqs:", recv_ans.reqs)
                for req_id in recv_ans.reqs:
                    try:
                        del self.req_id_to_out_inf[req_id]
                    except:
                        pass

        return
