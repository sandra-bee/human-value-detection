from datasets import load_dataset
import ast

# Imports dataset from huggingface
# Dataset imported: https://huggingface.co/datasets/webis/Touche23-ValueEval
# Usage code taken directly from there
def convert_labels(example):
    example["Labels"] = [i for i in ast.literal_eval(example["Labels"])]
    return example

valueeval23 = load_dataset("webis/Touche23-ValueEval")

# These datasets are in arrow format datasets
# https://huggingface.co/docs/datasets/about_arrow
training_dataset = valueeval23["training"].map(convert_labels)
validation_dataset = valueeval23["validation"].map(convert_labels)
validation_zhihu_dataset = valueeval23["validationzhihu"].map(convert_labels)

