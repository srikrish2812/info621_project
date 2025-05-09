"""
Task class for evaluating the GSM8K (Grade School Math 8K) dataset.

Unsloth Docs: https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo
"""
import pdb
import re
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from datasets import DatasetDict, Dataset, load_dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

SYSTEM_PROMPT2 = """
<start_of_turn>user
Solve the following math problem step-by-step:
{question}<end_of_turn>
<start_of_turn>model
"""

@dataclass
class GSM8kTask:
	"""
	Class for keeping track of experiments and results on the GSM8k dataset.

	Question (x)	: a grade school math word problem
	Output (y)		: a set of reasoning steps to solve the math problem;
	Reward (r)		: the reward function					  
	"""
	config_name: str = "main"

	def __len__(self, split="train"):
		return len(self.dataset[split])

	def __getitem__(self, idx, split="train"):
		return self.dataset[split][idx]

	def __post_init__(self):
		self.dataset: DatasetDict[str, Dataset] = load_dataset(
			"openai/gsm8k",
			self.config_name,
			revision="e53f048856ff4f594e959d75785d2c2d37b678ee")

	def __getsamples__(self, n_samples=10, split="train"):
		indices = np.random.randint(low=0, high=self.__len__(split=split), size=n_samples)

		return self.dataset_prompts[split].shuffle(seed=42).select(indices)

	# Unsloth
	def extract_xml(self, text: str) -> str:
		answer = text.split("<answer>")[-1]
		answer = answer.split("</answer>")[0]
		return answer.strip()

	# Unsloth
	def extract_hash_answer(self, text: str) -> str | None:
		if "####" not in text:
			return None
		return text.split("####")[1].strip()

	# Unsloth 
	def get_questions(self, split="train", prompt=SYSTEM_PROMPT2) -> Dataset:
		self.dataset_prompts = self.dataset.map(
			lambda x: {
				"prompt": [
					{"role": "system", "content": prompt},
					{"role": "user", "content": x["question"]},
				],
				"answer": self.extract_hash_answer(x["answer"]),
			}
		)
		return self.dataset_prompts

	def extract_answer(text):
		"""
		Extracts the last occurring number (i.e., answer) from a string.
		"""
		if not text:
			return None

		text = text.replace(',', '')
		numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)

		if numbers:
			return numbers[-1]

		return None


def main(test=False):
	if test:
		gsm8k = GSM8kTask()
		dataset = gsm8k.get_questions()
		sample = gsm8k.__getsamples__(n_samples=1, split="test")
		pdb.set_trace()


if __name__ == "__main__":
	main(test=True)
