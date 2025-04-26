from vllm import LLM, SamplingParams

llm = LLM(model="gpt2", swap_space=2)  # or any HuggingFace model
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate("What is the capital of France?", sampling_params)
response = outputs[0]
response_text = response.outputs[0].text
num_tokens = len(response.outputs[0].token_ids)
# print(outputs[0].outputs[0].text)
print(f"Number of response tokens: {num_tokens}")