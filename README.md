# Reinforcement Learning Evokes Reasoning in Small Language Models

Team Name: Digital Dragons

Contributors: Abhay Nandiraju, Ayesha Khatun, Maksim Kulik, Matthew Hernandez

> **Note** The scope of this project involves the following points: experiment with fine-tuning a reasoning model with reinforcement learning, track progress by evaluating on GSM8K benchmark, and assess the capability of reasoning in LLMs.

This project was created for the purpose of applying current techniques in the Machine Learning world to stay up-to-data with recent advancements. Are LLMs genuinely capable of true logical reasoning?

## Project Structure
```bash
	│── logs/
	│   ├── baseline/            # Logs for baseline model
	│   └── model_reasoning/     # Logs for model with reasoning
	│── src/
	│   ├── config/              # Configuration for models
	│   ├── notebooks/           # Code for experimnand exploratory data analysis
	│   └── tasks/               # Helper task for modular coding
	│── tests/                   # Unit & integration tests
	│── README.md                # Project overview and setup instructions
	│── project_charter.txt      # Short document stating project details
	│── requirements.txt         # Requirements for project

```

## Setup 
The code must be run with a GPU only because of Unsloth's requirements, and we recommend using Google Colab to run the inference notebook to run a selected sample from the dataset. The notebook has a direct link to a Colab instance where you can run the code easily.

## Quick Start
The following code is from the ```src/notebooks/gemma_3_final_inference.ipynb``` notebook and runs inference on a preselected example:

```Python
model_output = run_model(sample['prompt'], measure_p=False)
```
And the output would be something like the response below (note the LLM is not deterministic, and may output a incorrect response):
```Bash
<bos><bos><start_of_turn>user

<start_of_turn>user
Solve the following math problem step-by-step:
{question}<end_of_turn>
<start_of_turn>model


Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<end_of_turn>
<start_of_turn>model
Let $C$ be the number of clips Natalia sold in April.
Let $M$ be the number of clips Natalia sold in May.

Natalia sold clips to 48 of her friends in April, so she sold 48 clips in April.
So, $C = 48$.
In May, she sold half as many clips as in April. So, $M = \frac{1}{2}C = \frac{1}{2} \times 48 = 24$.
So, Natalia sold 24 clips in May.
The total number of clips sold in April and May is $C + M = 48 + 24 = 72$.

Thus, Natalia sold 72 clips altogether in April and May.

Final Answer: The final answer is $\boxed{72}$<end_of_turn>
```
