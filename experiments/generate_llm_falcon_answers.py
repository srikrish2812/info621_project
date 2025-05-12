import json
from transformers import pipeline

# Load dataset
with open("numaric_math_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load a small language model for CPU use (math-friendly)
generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device=-1)

# Extract answers using LLM
output_data = []
for item in data:
    question = item["question"]
      #  prompt = f"{question}\nAnswer with a number only:"
         # prompt = f"{question}\n Now, step by step follow this rules, 1. read the question, 2. calculate the answer and 3. Answer with a number only:"


    prompt = f"Sample Question is: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \nfor this question answer will be like that, Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72, now for this question {question}\n calculate the math Answer with a number only like after ####"
    response = generator(prompt, max_new_tokens=20, do_sample=False)[0]["generated_text"]
    
    # Extract the last number from the output
    import re
    match = re.findall(r"\d+", response.split("Answer with a number only:")[-1])
    num_answer = match[-1] if match else "0"

    output_data.append({
        "question": question,
        "numaric_answer": num_answer,
        "response": response
    })

# Save generated answers
with open("llm_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

    ## pip install transformers  
    # pip install torch
#to run this code
