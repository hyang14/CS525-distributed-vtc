import time
import requests
from server.load_balancer_client import Client 

# Define configuration (adjust IPs and ports)
config = {
    "max_probe_pool_size": 16,
    "num_replicas": 3,
    "probe_rate": 1.0,
    "q_rif_threshold": 0.6,
    "delta_reuse": 0.1,
    "max_probe_age": 5,
    "max_probe_use": 3,
    "servers": [
        "sp25-cs525-0806.cs.illinois.edu:8001",
        "sp25-cs525-0807.cs.illinois.edu:8002"
    ]
}

# Initialize the client
client = Client(config=config, servers=config["servers"], mode="hcl")
print("Client initialized. Probing servers...")

# Let probes accumulate
time.sleep(5)  # Give it 5 seconds to collect probe info


print("=== Warm-up Phase: Sending 1 ping to each server ===")
for s in config["servers"]:
    try:
        r = requests.get(f"http://{s}/ping")
        print(f"Warm-up success for {s}")
    except Exception as e:
        print(f"Warm-up failed for {s}: {e}")

# Run a few test dispatches
try:
    print("=== Test: Ping ===")
    client.send_ping()
    print("Ping successful")

    print("=== Test: Medium ===")
    for _ in range(5):
        client.send_medium()
        time.sleep(1)
        print("Medium process successful")

    print("=== Test: Batch ===")
    client.send_batch(["test1", "test2"])
    print("Batch process successful")

except Exception as e:
    print("Error during request:", e)

# Optional: Show current probe pool
print("\n=== Probe Pool ===")
for probe in client.probes:
    print(f"Server: {probe.server_id}, RIF: {probe.rif}, Latency: {probe.latency:.3f}, Normalized RIF: {probe.normalized_rif:.2f}")

# Stop client
client.stop()
print("Client stopped.")
