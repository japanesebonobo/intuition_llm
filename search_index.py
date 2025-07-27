import faiss
import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm

api_key='vc9NrBvvqMtV48mEFZ9NhqGaERdLAaRgNWp3m2CY'

co = cohere.Client(api_key)

def search(query, number_of_results=3):
    query_embed = co.embed(texts=[query], input_type="search_query",).embeddings[0]

    distances, similar_item_ids = index.search(np.float32([query_embed]), number_of_results)
    texts_np = np.array(texts)
    results = pd.DataFrame(data={
        "text": texts_np[similar_item_ids[0]],
        "distance": distances[0],
    })

    print(f"Query: '{query}'\nNearest neighbors:")
    return results

dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
print(index.is_trained)
index.add(np.float32(embeds))

query = "How pricise was the science"
results = search(query)
print(results)
