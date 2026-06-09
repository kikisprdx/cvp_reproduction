from datasets import ClassLabel, Value, load_dataset


def load_data():
    dataset = load_dataset("robro/cifar10-c-parquet", split="train")
    dataset = dataset.sort("corruption_name")
    return dataset
