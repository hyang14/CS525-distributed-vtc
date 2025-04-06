from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")  # or any HuggingFace model
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate("What is the capital of France?", sampling_params)
print(outputs[0].outputs[0].text)