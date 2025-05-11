test_data.json --> this file taken from huggingface GSM8K dataset (test file)
comma.py --> code for adding comma to make editable
commaadd.json ---> json file after adding modification of orginal dataset file
numaric_answer.py --> for add new colunm name numaric answer to test data
numaric_math_data.json --> final data after preprocessing
generate_llm_falcon_answers.py  --> generated data using falcon pretrained model
evaluate_llm_answers.py  --> evaluate the accuracy (will change the jeson file for each and every llm model)
llm_dataset.json  --> this json file contain the generated output by llm, here showing the data from falcon pretrained 1 B model
