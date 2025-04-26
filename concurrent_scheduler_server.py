import asyncio
import httpx
import random
import uuid

SUBMIT_URL = "http://localhost:5000/submit"
DISPATCH_URL = "http://localhost:5000/dispatch"

users = ["alice", "bob", "charlie", "dave"]
prompts = [
    "What is the capital of France?",
    "Tell me a joke.",
    "What is 9 + 10?",
    "Describe the water cycle.",
    "What's the speed of light?",
    "Explain Newton's laws.",
    "Translate 'hello' to Spanish.",
    "Who won the World Cup in 2018?",
]
# prompts = [
#     "What is the capital of France?",
#     "Tell me a joke.",
#     "What is 9 + 10?",
#     "Describe the water cycle.",
#     "What's the speed of light?",
#     "Explain Newton's laws.",
#     "Translate 'hello' to Spanish.",
#     "Who won the World Cup in 2018?",
#     "What causes rainbows?",
#     "What's the tallest mountain on Earth?",
#     "Name the planets in our solar system.",
#     "What is quantum physics?",
#     "Who wrote 'Romeo and Juliet'?",
#     "How many continents are there?",
#     "What's the boiling point of water?",
#     "Who painted the Mona Lisa?",
#     "Define photosynthesis.",
#     "What is machine learning?",
#     "How do airplanes fly?",
#     "What does DNA stand for?",
#     "What's the currency of Japan?",
#     "How old is the Earth?",
#     "What's the largest mammal?",
#     "Explain gravity in simple terms.",
#     "What is blockchain technology?",
#     "Name the three states of matter.",
#     "Who is the CEO of Tesla?",
#     "What is the capital of Brazil?",
#     "How do vaccines work?",
#     "What's the square root of 144?",
#     "Name five programming languages.",
#     "What's the chemical symbol for gold?",
#     "What is global warming?",
#     "How does the internet work?",
#     "Translate 'good night' to French.",
#     "What is the Pythagorean theorem?",
#     "What language is spoken in Egypt?",
#     "What's the formula for force?",
#     "What is artificial intelligence?",
#     "Who discovered electricity?",
#     "What's the difference between a comet and an asteroid?",
#     "How do plants make food?",
#     "What is a black hole?",
#     "Who was the first person on the moon?",
#     "What is the meaning of life?",
#     "What is climate change?",
#     "What's the longest river in the world?",
#     "What does HTTP stand for?",
#     "Explain how tides work.",
#     "What are the primary colors?",
# ]


async def submit_prompt(user_id, prompt, client):
    payload = {
        "user_id": user_id,
        "prompt": prompt,
        "max_tokens": 20
    }
    try:
        response = await client.post(SUBMIT_URL, json=payload)
        print(f" Submitted from {user_id}: {response.json()}")
    except Exception as e:
        print(f" Submit failed for {user_id}: {e}")


async def dispatch_batch(client):
    try:
        response = await client.post(DISPATCH_URL)
        print(f" Dispatch result: {response.json()}")
    except Exception as e:
        print(f" Dispatch failed: {e}")


async def simulate_users(n_requests=20, dispatch_interval=4):
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = []

        for i in range(n_requests):
            user = random.choice(users)
            prompt = random.choice(prompts)
            task = submit_prompt(user, f"{prompt} [req {i}]", client)
            tasks.append(task)

            # Periodically dispatch
            if i > 0 and i % dispatch_interval == 0:
                tasks.append(dispatch_batch(client))

            await asyncio.sleep(0.1)  # simulate arrival delay

        await asyncio.gather(*tasks)

        # Final dispatch
        await dispatch_batch(client)


if __name__ == "__main__":
    asyncio.run(simulate_users(n_requests=32, dispatch_interval=6))

