from datasets import load_dataset, Value, ClassLabel

def load_data():
    dataset = load_dataset("robro/cifar10-c-parquet", split="train")
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    return dataset
