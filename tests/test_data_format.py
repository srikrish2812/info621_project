import pytest
import random
import re
import datasets

@pytest.fixture(scope="module")
def hf_sample():
	gsm8k = datasets.load_dataset("openai/gsm8k", "main", split="train")
	idx = random.randint(0, len(gsm8k) - 1)
	sample = gsm8k[idx]
	sample['idx'] = idx

	return sample

def extract_answer(answer):
	try:
		return re.split(r"####\s*", answer)[-1].strip()
	except Exception as e:
		print(f"Warning: Could not parse gold answer. \nError: {e}")
		return None

@pytest.mark.parametrize(
	"test_input,expected", [("Explanation. #### 7", "7"), ("Explanation. 7", "Explanation. 7")])
def test_gold_answer(test_input, expected, hf_sample):
	answer_full = hf_sample['answer'].strip()
	assert extract_answer(test_input) == expected
	assert isinstance(extract_answer(answer_full), (str, float, int))

def test_question_formatting(hf_sample):
	question = hf_sample['question']
	prompt = f"<start_of_turn>user\nSolve the following math problem step-by-step:\n{question}<end_of_turn>\n<start_of_turn>model\n"
	assert len(prompt) > len(question)

