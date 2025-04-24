"""
Task class for evaluating the GSM8K (Grade School Math 8K) dataset.
"""
import pdb
from dataclasses import dataclass
from typing import TypedDict

import asteval
import transformers
import numpy as np
from datasets import DatasetDict, Dataset, load_dataset


@dataclass
class GSM8kTask:
	"""
	Class for keeping track of experiments and results on the GSM8k dataset.

	Question (x)	: a grade school math word problem
	Output (y)		: a set of reasoning steps to solve the math problem;
	Reward (r)		: the reward function					  
	"""
	config_name: str = "main"

	def __len__(self):
		return len(self.dataset['train'])

	def __getitem__(self, idx, split="train"):
		return self.dataset[split][idx]

	def __post_init__(self):
		self.dataset: DatasetDict[str, Dataset] = load_dataset("openai/gsm8k", self.config_name)
		self.interpreter = asteval.Interpreter()

	def __getsamples__(self, n_samples=10, split="train"):
		indices = np.random.randint(low=0, high=self.__len__(), size=n_samples)
		return self.dataset[split].shuffle(seed=42).select(indices)

	def get_input(self):
		pass

	def get_output(self):
		pass

	def run_experiment(self):
		pass 


def main():
	gsm8k = GSM8kTask()
	pdb.set_trace()


if __name__ == "__main__":
	main()
