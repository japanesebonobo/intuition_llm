from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

vectors = model.encode("Best movie ever!", convert_to_tensor=True)

print(vectors.shape)  # Should print: torch.Size([768])
