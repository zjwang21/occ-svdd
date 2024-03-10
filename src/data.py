from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
import torch
import os

def get_dataset(path, min_value=None, max_value=None):
    data = torch.load(path, map_location='cpu')
    if min_value == None:
        min_value = data.min().item()
    if max_value == None:
        max_value = data.max().item()
    instance = max_value - min_value

    def preprocess_func(samples):
        samples = torch.FloatTensor(samples['point'])
        samples = (samples - min_value) / instance
        return {"normalized_point": [k for k in samples]}

    dataset = Dataset.from_dict({"point": [k for k in data]})
    dataset = dataset.map(preprocess_func, batched=True, remove_columns=dataset.column_names)
    print(dataset)
    return dataset, min_value, max_value

def get_test_dataset(path, min_value, max_value):
    dataset = load_from_disk(path)
    min_value = min()
    instance = max_value - min_value
    def preprocess_func(samples):
        temp = torch.FloatTensor(samples['point'])
        temp = (temp - min_value) / instance
        return {"normalized_point": [k for k in temp], "idx": samples['idx'], "labels": samples['labels']}

    dataset = dataset.map(preprocess_func, batched=True, remove_columns=dataset.column_names)
    print(dataset)
    return dataset