from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed

model_id = "meta-llama/Llama-3.1-8B-Instruct"

ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {"enabled": False},
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 3},  # Fully Sharded Data Parallel (FSDP)
    "zero_allow_untested_optimizer": True,
    "offload_optimizer": {"device": "cpu"}
}

deepspeed.init_distributed()

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")

model = deepspeed.initialize(model=model, config=ds_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "What is AI?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.module.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
