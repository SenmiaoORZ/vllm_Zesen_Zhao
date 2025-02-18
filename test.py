import torch
from vllm import LLM

def log_vram_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
    print(f"Allocated VRAM: {allocated:.2f} MB, Reserved VRAM: {reserved:.2f} MB")

prompt = "Hello"

# Log VRAM before model initialization
log_vram_usage()

llm_b = LLM(
    model="gpt2",task="generate"
)

# Log VRAM after model initialization
log_vram_usage()

outputs_b = llm_b.generate(prompt)[0].outputs[0].text

# Log VRAM after inference
log_vram_usage()

print(prompt, outputs_b)