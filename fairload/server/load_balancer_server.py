from flask import Flask, request, jsonify
import threading
import time
import random
import sys
from collections import deque

app = Flask(__name__)

# ---- In-Memory RIF-Latency Tracker ----
class Metric:
    def __init__(self, rif, latency):
        self.rif = rif
        self.latency = latency

class MetricReporter:
    def __init__(self, max_metrics=1000):
        self.metrics = deque(maxlen=max_metrics)
        self.lock = threading.Lock()

    def record(self, rif, latency):
        with self.lock:
            self.metrics.append(Metric(rif, latency))

    def get_nearest_latencies(self, rif):
        with self.lock:
            if not self.metrics:
                return 0.0
            sorted_by_diff = sorted(self.metrics, key=lambda m: abs(m.rif - rif))
            top_k = sorted_by_diff[:5]
            latencies = sorted([m.latency for m in top_k])
            return latencies[len(latencies) // 2]

metric_reporter = MetricReporter()
rif_counter = 0
rif_lock = threading.Lock()

# ---- Helpers for RIF ----
def increment_rif():
    global rif_counter
    with rif_lock:
        rif_counter += 1
        return rif_counter

def decrement_rif():
    global rif_counter
    with rif_lock:
        rif_counter -= 1
        return rif_counter

def get_current_rif():
    with rif_lock:
        return rif_counter

# ---- Route Handlers ----
@app.route("/batch", methods=["POST"])
def batch():
    rif = increment_rif()
    start = time.time()
    try:
        data = request.get_json()
        strings = data.get("strings", [])
        base = 10
        offset = random.randint(-5, 5)
        time.sleep(base + offset)  # simulate processing time
        return jsonify({"message": f"Processed batch of {strings}"})
    finally:
        duration = time.time() - start
        decrement_rif()
        metric_reporter.record(rif, duration)

@app.route("/ping", methods=["GET"])
def ping():
    rif = increment_rif()
    start = time.time()
    try:
        return jsonify({"message": "pong"})
    finally:
        duration = time.time() - start
        decrement_rif()
        metric_reporter.record(rif, duration)

@app.route("/medium", methods=["POST"])
def medium():
    rif = increment_rif()
    start = time.time()
    try:
        base = 3
        offset = random.randint(-1, 1)
        time.sleep(base + offset)
        return jsonify({"message": "Medium process complete"})
    finally:
        duration = time.time() - start
        decrement_rif()
        metric_reporter.record(rif, duration)

@app.route("/probe", methods=["GET"])
def probe():
    time.sleep(0.01)
    rif = get_current_rif()
    median_latency = metric_reporter.get_nearest_latencies(rif)
    return jsonify({
        "rif": rif,
        "latency": median_latency
    })

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app.run(host="0.0.0.0", port=port)
