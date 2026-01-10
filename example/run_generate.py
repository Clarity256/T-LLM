import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/data/models/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="cuda:0"
).eval()

messages = [{"role":"user", "content": "用一句话解释什么是KV Cache"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False   # 先关闭thinking
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, do_sample=False, max_new_tokens=64)

print(tokenizer.decode(out[0], skip_special_tokens=True))
