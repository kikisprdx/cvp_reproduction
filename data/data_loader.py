from datasets import load_dataset


def load_data():
    dataset = load_dataset(
        "randall-lab/cifar10-c", split="test", trust_remote_code=True
    )

    example = dataset[0]
    image = example["image"]
    label = example["label"]

    image.show()
    print(f"label", label)


if __name__ == "main":
    load_data()
