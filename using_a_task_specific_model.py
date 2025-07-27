from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report

def evaluate_model(y_true, y_pred):
    perfomance = classification_report(
        y_true,
        y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(perfomance)


model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device="cuda:0"  # Use "cpu" if you don't have a GPU
)

y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"],"text")), total=len(data["test"])):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)
    

evaluate_model(data["test"]["label"], y_pred)
