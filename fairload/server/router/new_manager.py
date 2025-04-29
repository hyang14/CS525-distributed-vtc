"""
RouterManagerHTTP – streaming scheduler-aware router for external LLM engines
============================================================================
Keeps *the same fairness‐scheduler layer* (ReqQueue & friends) from the original
implementation while forwarding prompts to an HTTP‑based engine that speaks the
OpenAI streaming (`data:`) protocol.

Design overview
---------------
* **HttpServerManager**  → ZMQ **PULL** → `RouterManagerHTTP`
* Router retains *get_scheduler* + ReqQueue to decide **when** a request is
  allowed to start based on token budgets / fairness weights.
* Up to `max_concurrency` requests run *concurrently*; each maps 1‑to‑1 onto an
  async HTTP connection.
* Tokens from the engine are streamed back to **HttpServerManager** via a ZMQ
  **PUSH** socket as `BatchStrOut` objects (exactly what it already expects).
* Detokenizer process is **gone**.

Only external dependency added is **aiohttp** for async HTTP.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import aiohttp
import uvloop
import zmq
import zmq.asyncio

# Scheduler bits – reused from the original code base
from ..input_params import InputParams  # type: ignore
from ..io_struct import AbortReq, BatchStrOut, Req  # type: ignore
from ..router.req_queue import ReqQueue  # base class – others subclass it  # type: ignore
# from ..router.manager import get_scheduler  # we can reuse helper  # type: ignore

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


# ---------------------------------------------------------------------------
# RouterManagerHTTP
# ---------------------------------------------------------------------------


class RouterManagerHTTP:
    """Scheduler‑aware router that forwards prompts to an external engine."""

    # "How many engine connections at once?"  In fair‑scheduling mode you
    # normally bound total *tokens* rather than requests; we approximate with a
    # simple concurrent‑request cap, default 32.
    max_concurrency = int(os.environ.get("SLORA_HTTP_MAX_CONCURRENCY", 32))

    def __init__(
        self,
        *,
        serving_engine_url: str,
        router_port: int,
        httpserver_port: int,
        adapter_dirs: list[str],
        input_params: InputParams,
        eos_id: int = 2,
    ) -> None:
        self.serving_engine_url = serving_engine_url.rstrip("/")
        self.eos_id = eos_id
        self.input_params = input_params

        # ZMQ wiring
        ctx = zmq.asyncio.Context(2)
        self.recv_from_httpserver = ctx.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        self.send_to_httpserver = ctx.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")

        # Fairness scheduler (unchanged from original)
        self.req_queue: ReqQueue = get_scheduler(input_params, adapter_dirs)
        self.lora_ranks: dict[str | None, int] = {None: 0}

        # Async HTTP + task bookkeeping
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_counters: Dict[str, int] = {}
        self._inflight_tasks: Dict[str, asyncio.Task] = {}

        # Events
        self._new_req_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public lifecycle helpers
    # ------------------------------------------------------------------

    async def start(self) -> None:  # entry point
        self.session = aiohttp.ClientSession()
        await asyncio.gather(
            self._consumer_loop(),  # ingest ZeroMQ requests / aborts
            self._scheduler_loop(),  # fairness scheduling & task launch
        )

    async def close(self) -> None:  # just in case
        if self.session:
            await self.session.close()

    # ------------------------------------------------------------------
    # Consumer – receives requests / aborts from HttpServerManager
    # ------------------------------------------------------------------

    async def _consumer_loop(self) -> None:
        while True:
            msg = await self.recv_from_httpserver.recv_pyobj()

            if isinstance(msg, AbortReq):
                await self._handle_abort(msg.req_id)
                continue

            # Unpack new request tuple from HttpServerManager
            adapter_dir, prompt_ids, sampling_params, request_id = msg  # type: ignore[arg-type]

            # Wrap into Req object so ReqQueue can reason about lengths
            req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
            self.req_queue.append(req)

            # Wake up scheduler loop
            self._new_req_event.set()

    # ------------------------------------------------------------------
    # Scheduler loop – decides when to launch new requests
    # ------------------------------------------------------------------

    async def _scheduler_loop(self) -> None:
        while True:
            # Wait until either a running slot opens or a new request arrives
            if (
                len(self._inflight_tasks) >= self.max_concurrency
                or self.req_queue.is_empty()
            ):
                self._new_req_event.clear()
                await self._new_req_event.wait()
                continue

            # Fair scheduler selects next request(s)
            next_batch = self.req_queue.generate_new_batch(
                running_batch=None, lora_ranks=self.lora_ranks
            )
            if next_batch is None:
                # Probably token budget exhausted; wait for completions
                self._new_req_event.clear()
                await self._new_req_event.wait()
                continue

            # Launch each req in the selected batch
            for req in next_batch.reqs:
                self._launch_request_task(req)

    # ------------------------------------------------------------------
    # Launch + streaming handling for a single request
    # ------------------------------------------------------------------

    def _launch_request_task(self, req: Req) -> None:
        self._inflight_tasks[req.request_id] = asyncio.create_task(
            self._forward_to_engine(req)
        )

    async def _forward_to_engine(self, req: Req) -> None:
        """Stream from engine, relay tokens, then notify scheduler."""
        prompt = self._decode_prompt_ids(req.prompt_ids)
        payload = {
            "model": req.adapter_dir if req.adapter_dir else "default",
            "prompt": prompt,
            "max_tokens": req.sampling_params.max_new_tokens,
            "temperature": req.sampling_params.temperature,
            "top_p": req.sampling_params.top_p,
            "stream": True,
        }
        url = f"{self.serving_engine_url}/v1/completions"

        try:
            assert self.session is not None
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    await self._finish_with_error(req.request_id, f"HTTP {resp.status}")
                    return
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line.startswith(b"data: "):
                        continue
                    chunk = line[6:]
                    if chunk == b"[DONE]":
                        break
                    try:
                        data_json = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    await self._relay_token(req.request_id, data_json)
        except asyncio.CancelledError:
            await self._finish_with_error(req.request_id, "aborted")
            raise
        except Exception as exc:
            await self._finish_with_error(req.request_id, str(exc))
        finally:
            await self._finalize_request(req)

    # ------------------------------------------------------------------
    # Helpers: relay token / abort / error / finalize
    # ------------------------------------------------------------------

    async def _relay_token(self, request_id: str, data_json: Dict[str, Any]) -> None:
        choices = data_json.get("choices", [])
        if not choices:
            return
        token_text = choices[0].get("text", "")
        finish_reason = choices[0].get("finish_reason")
        token_id = self._token_counters.get(request_id, 0) + 1
        self._token_counters[request_id] = token_id
        metadata = {"id": token_id, "text": token_text, "logprob": None}
        finished = finish_reason is not None

        batch = BatchStrOut()
        batch.reqs_infs.append((request_id, token_text, metadata, finished, False))
        self.send_to_httpserver.send_pyobj(batch)

    async def _handle_abort(self, request_id: str) -> None:
        task = self._inflight_tasks.pop(request_id, None)
        if task:
            task.cancel()
        await self._finish_with_error(request_id, "aborted by client")

    async def _finish_with_error(self, request_id: str, err: str) -> None:
        batch = BatchStrOut()
        batch.reqs_infs.append((request_id, "", {"error": err}, True, True))
        self.send_to_httpserver.send_pyobj(batch)

    async def _finalize_request(self, req: Req) -> None:
        # Remove from inflight, free token counter, nudge scheduler
        self._inflight_tasks.pop(req.request_id, None)
        self._token_counters.pop(req.request_id, None)
        self.req_queue.finish_req(req)
        self._new_req_event.set()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_prompt_ids(prompt_ids: Any) -> str:  # noqa: ANN401
        try:
            return " ".join(map(str, prompt_ids))
        except Exception:
            return str(prompt_ids)


# ---------------------------------------------------------------------------
# Sub‑process entry point – launched from api_server.py
# ---------------------------------------------------------------------------

def start_router_process(
    args,
    router_port: int,
    httpserver_port: int,
    pipe_writer,
):
    async def _runner() -> None:
        try:
            input_params = InputParams(
                max_req_total_len=args.max_req_total_len,
                max_total_token_num=args.max_total_token_num,
                batch_max_tokens=args.batch_max_tokens,
                running_max_req_size=args.running_max_req_size,
                scheduler=args.scheduler,
                fair_weights=args.fair_weights,
            )
            router = RouterManagerHTTP(
                serving_engine_url=args.serving_engine_url,
                router_port=router_port,
                httpserver_port=httpserver_port,
                adapter_dirs=args.lora_dirs,
                input_params=input_params,
                eos_id=args.eos_id,
            )
            pipe_writer.send("init ok")
            await router.start()
        except Exception as exc:
            import traceback

            pipe_writer.send("\n".join(traceback.format_exception(exc)))
            raise

    asyncio.run(_runner())
