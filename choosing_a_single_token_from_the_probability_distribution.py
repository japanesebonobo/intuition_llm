import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    trust_remote_code=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False
)

prompt = "The capital of France is"

input_ids = tokenizer(
    prompt,
    return_tensors="pt"
).input_ids.to("cuda")

model_outputs = model.model(input_ids=input_ids)

print(model_outputs[0].shape)  # (batch_size, sequence_length, vocab_size)

lm_head_output = model.lm_head(model_outputs[0])

print(lm_head_output.shape)  # (batch_size, sequence_length, vocab_size)

token_id = lm_head_output[0, -1].argmax(-1)

print(tokenizer.decode(token_id))
