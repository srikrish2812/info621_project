import json
from sklearn.metrics import accuracy_score

# Load both datasets
with open("numaric_math_data.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

with open("llm_dataset.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# Match and evaluate
true_vals, pred_vals = [], []

for gt, pred in zip(ground_truth, predictions):
    if gt["question"].strip() == pred["question"].strip():
        try:
            true_vals.append(int(gt["numaric_answer"]))
            pred_vals.append(int(pred["numaric_answer"]))
        except:
            continue

# Calculate accuracy
accuracy = accuracy_score(true_vals, pred_vals)

# Print results
print(f"Accuracy: {accuracy:.4f}")

