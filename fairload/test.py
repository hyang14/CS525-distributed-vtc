import requests

servers = ["sp25-cs525-0805.cs.illinois.edu:8000", "sp25-cs525-0806.cs.illinois.edu:8001", "sp25-cs525-0807.cs.illinois.edu:8002"]

for server in servers:
    print(f"--- Testing {server} ---")
    
    # Test ping
    try:
        r = requests.get(f"http://{server}/ping")
        print("Ping:", r.json())
    except Exception as e:
        print("Ping failed:", e)

    # Test medium
    try:
        r = requests.post(f"http://{server}/medium")
        print("Medium:", r.json())
    except Exception as e:
        print("Medium failed:", e)

    # Test batch
    try:
        r = requests.post(f"http://{server}/batch", json={"strings": ["hello", "world"]})
        print("Batch:", r.json())
    except Exception as e:
        print("Batch failed:", e)

    # Test probe
    try:
        r = requests.get(f"http://{server}/probe")
        print("Probe:", r.json())
    except Exception as e:
        print("Probe failed:", e)
