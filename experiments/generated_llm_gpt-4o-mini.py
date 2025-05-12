import json
import re
from openai import OpenAI
from sklearn.metrics import accuracy_score

client = OpenAI(api_key="sk-xxxxxxx")  # I have used my API

with open("numaric_math_data.json", "r", encoding="utf-8") as f:
    MATH_DATASET = json.load(f)

def generate_math_response(question):
    """Generate a step-by-step math solution using GPT-4o"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert math tutor. Solve problems step-by-step and provide the final numerical answer after '####'."},
                {"role": "user", "content": f"Solve: {question}\nShow your work step by step and provide the final answer after '####'."}
            ],
            temperature=0.3,
            max_tokens=256
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def extract_answer(response_text):
    """Extract the numerical answer from the response"""
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    numbers = re.findall(r"[-+]?\d*\.?\d+", response_text)
    return float(numbers[-1]) if numbers else None

def evaluate_math_reasoning(dataset, sample_size=None):
    """Evaluate the math reasoning performance"""
    if sample_size:
        dataset = dataset[:sample_size]
    
    predictions = []
    true_answers = []
    results = []
    
    for item in dataset:
        question = item["question"]
        true_answer = item["answer"]
        
        print(f"\nQuestion: {question}")
        print(f"Expected Answer: {true_answer}")
        
        response = generate_math_response(question)
        if not response:
            print("Failed to generate response")
            continue
            
        pred_answer = extract_answer(response)
        predictions.append(pred_answer)
        true_answers.append(true_answer)
        
        result = {
            "question": question,
            "response": response,
            "predicted": pred_answer,
            "actual": true_answer,
            "correct": abs(pred_answer - true_answer) < 0.01 if isinstance(true_answer, float) else pred_answer == true_answer
        }
        results.append(result)
        
        print(f"GPT-4o Response:\n{response}")
        print(f"Extracted Answer: {pred_answer}")
        print(f"Correct: {result['correct']}")
    
    # Calculate accuracy
    accuracy = accuracy_score(
        [abs(p - t) < 0.01 if isinstance(t, float) else p == t for p, t in zip(predictions, true_answers)],
        [True] * len(true_answers)
    )
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    return results, accuracy

# Run evaluation
results, accuracy = evaluate_math_reasoning(MATH_DATASET)

# Save results to JSON
with open("math_reasoning_results.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "results": results,
        "dataset": MATH_DATASET
    }, f, indent=2)

print("\nEvaluation complete. Results saved to math_reasoning_results.json")