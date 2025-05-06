import requests
import json
import threading
import time
import logging
from typing import List, Optional, Dict

class ProbeInfo:
    def __init__(self, rif, latency, server_id, timestamp, use_count=0):
        self.rif = rif
        self.latency = latency
        self.server_id = server_id
        self.timestamp = timestamp
        self.use_count = use_count
        self.normalized_rif = 1.0

class Client:
    def __init__(self, config: Dict, servers: List[str], mode: str = "hcl"):
        self.config = config
        self.servers = servers[:5]  # Limit to 5 servers
        self.mode = mode
        self.probes: List[ProbeInfo] = []
        self.max_rif = 0
        self.rr_index = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger("Client")
        self.logger.setLevel(logging.INFO)
        self.stop_flag = threading.Event()

        interval = 1.0 / self.config.get("probe_rate", 1)
        threading.Thread(target=self._probe_loop, args=(interval,), daemon=True).start()

    def _probe_loop(self, interval):
        while not self.stop_flag.is_set():
            self.probe()
            time.sleep(interval)

    def stop(self):
        self.stop_flag.set()

    def update_rif_distribution(self, probe: ProbeInfo):
        probe.normalized_rif = float(probe.rif) / self.max_rif if self.max_rif > 0 else 1.0

    def is_probe_hot(self, probe: ProbeInfo) -> bool:
        return probe.normalized_rif >= self.config.get("q_rif_threshold", 0.5)

    def select_replica(self, job_type: str) -> Optional[str]:
        with self.lock:
            if self.mode == "round_robin":
                return self._select_rr()
            return self._select_hcl()

    def _select_rr(self) -> str:
        if not self.servers:
            return None
        server = self.servers[self.rr_index]
        self.rr_index = (self.rr_index + 1) % len(self.servers)
        return server

    def _select_hcl(self) -> Optional[str]:
        print("\n[DEBUG] Current Probe Pool:")
        for p in self.probes:
            print(f"  Server: {p.server_id}, RIF: {p.rif}, Latency: {p.latency:.3f}, "
                  f"NormRIF: {p.normalized_rif:.2f}, UseCount: {p.use_count}, "
                  f"{'HOT' if self.is_probe_hot(p) else 'COLD'}")
        cold = [p for p in self.probes if not self.is_probe_hot(p)]
        hot = [p for p in self.probes if self.is_probe_hot(p)]

        if cold:
            selected = min(cold, key=lambda p: p.latency)
        elif hot:
            selected = min(hot, key=lambda p: p.rif)
        else:
            return None

        selected.use_count += 1
        return selected.server_id

    def probe(self):
        with self.lock:
            self._remove_stale_probes()
            for server in self.servers:
                try:
                    resp = requests.get(f"http://{server}/probe")
                    data = resp.json()
                    rif = data["rif"]
                    latency = data.get("latency", 0.0)
                    if rif > self.max_rif:
                        self.max_rif = rif
                    probe = ProbeInfo(rif, latency, server, time.time())
                    self.update_rif_distribution(probe)
                    self.probes.append(probe)
                except Exception as e:
                    self.logger.warning(f"Probe failed for {server}: {e}")

    def _remove_stale_probes(self):
        max_age = self.config.get("max_probe_age", 5)
        max_use = self.config.get("max_probe_use", 3)
        now = time.time()
        self.probes = [
            p for p in self.probes
            if now - p.timestamp < max_age and p.use_count < max_use
        ]

    def send_batch(self, strings: List[str]):
        server = self.select_replica("batch")
        if not server:
            raise Exception("No replica available")
        payload = {"strings": strings}
        resp = requests.post(f"http://{server}/batch", json=payload)
        if resp.status_code != 200:
            raise Exception(f"Server error: {resp.status_code}")

    def send_ping(self):
        server = self.select_replica("ping")
        if not server:
            raise Exception("No replica available")
        resp = requests.get(f"http://{server}/ping")
        if resp.status_code != 200:
            raise Exception(f"Server error: {resp.status_code}")

    def send_medium(self):
        server = self.select_replica("medium")
        if not server:
            raise Exception("No replica available")
        resp = requests.post(f"http://{server}/medium", json={})
        if resp.status_code != 200:
            raise Exception(f"Server error: {resp.status_code}")
