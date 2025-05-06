import pytest
import datasets

@pytest.fixture(scope="module")
def hf_data():
	gsm8k = datasets.load_dataset("openai/gsm8k", "main")
	return gsm8k

def test_load_data(hf_data):
	dummy_dict = {"train": None, "test": None}

	assert hf_data.keys() == dummy_dict.keys(), "The dataset should include ['train', 'test']"
	assert hf_data['train'].shape == (7473, 2), "The shape of TRAIN should be (7473, 2)"
	assert hf_data['test'].shape == (1319, 2), "The shape of TEST should be (1319, 2)"

def test_attributes(hf_data):
	expected_columns = ["question", "answer"]

	assert hf_data['train'].column_names == expected_columns
	assert hf_data['test'].column_names == expected_columns

def test_first_sample(hf_data):
	sample = hf_data['train'][0]

	assert isinstance(sample['question'], str)
	assert isinstance(sample['answer'], str)
	assert len(sample['question']) > 0
	assert len(sample['answer']) > 0
