from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-base",
    trust_remote_code=False,
)

model = AutoModel.from_pretrained(
    "microsoft/deberta-v3-xsmall",
    trust_remote_code=False,
)

tokens = tokenizer(
    "Hello world",
    return_tensors="pt",
)

outputs = model(**tokens)[0]

for token in tokens.input_ids[0]:
    print(tokenizer.decode(token))
