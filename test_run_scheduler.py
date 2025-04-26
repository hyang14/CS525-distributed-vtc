import requests

resp = requests.post("http://localhost:5000/submit", json={
    "user_id": "alice",
    "prompt": "What is the capital of France?",
    "max_tokens": 50
})

print("Status code:", resp.status_code)
print("Raw text:", resp.text)