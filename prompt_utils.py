import pandas as pd
import datasets
from datasets import Dataset

def formatting_prompts_func(examples):
    instruction =  "Given the following background and objective sentence(s) extracted from a scientific abstract, generate sentences pertaining to the abstract's methods and results."
    inputs = examples["background"]
    outputs = examples["target"]
 
    texts = []
    for input, output in zip(inputs, outputs):

        context_str = ""

        text = '''You are an AI assistant specializing in materials science hypothesis generation. Your task is to output a materials science hypothesis in the form of a paragraph describing methods and results given a research statement containing background and objective information.
        Now, carefully consider the following research statement:
        {}
        Your goal is to generate a hypothesis in the form of a paragraph that includes methods and expected results that logically follow from the provided research statement.
        Ensure that your final hypothesis maintains a high standard of novelty, scientific accuracy, intellectual stimulation, practical utility to researchers, and detail.
        Include only your final hypothesis in your output.
        Final Hypothesis:
        {}'''.format(input, output)

        text += '<|end_of_text|>' 
        texts.append(text)
    return {"text": texts,}

def formatting_prompts_func_test(examples):
    #instructions = examples["instruction"]
    instruction =  "Given the following background and objective sentence(s) extracted from a scientific abstract, generate sentences pertaining to the abstract's methods and results."
    inputs = examples["background"]
    outputs = examples["target"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = '''You are an AI assistant specializing in materials science hypothesis generation. Your task is to output a materials science hypothesis in the form of a paragraph describing methods and results given a research statement containing background and objective information.
        Now, carefully consider the following research statement:
        {}
        Your goal is to generate a hypothesis in the form of a paragraph that includes methods and expected results that logically follow from the provided research statement.
        Ensure that your final hypothesis maintains a high standard of novelty, scientific accuracy, intellectual stimulation, practical utility to researchers, and detail.
        Include only your final hypothesis in your output.
        Final Hypothesis:
        {}'''.format(input, '')
        texts.append(text)
    return {"text": texts,}


def get_datasets(train_data_file, test_data_file):
    train = pd.read_json(path_or_buf=train_data_file, lines=True)
    train_dataset = Dataset.from_pandas(train)
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    print(train_dataset[0:5])

    test = pd.read_json(path_or_buf=test_data_file, lines=True)
    test_dataset = Dataset.from_pandas(test)
    test_dataset = test_dataset.map(formatting_prompts_func_test, batched=True)

    return train_dataset, test_dataset
