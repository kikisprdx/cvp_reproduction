# Overview of dataset
# %%[jupynvim:cell-sep]
import sys, os 
sys.path.insert(0, os.path.abspath(".."))
# %%[jupynvim:cell-sep]
import datasets.config
import PIL
import pandas as pd
from data.data_loader import get_data_loader_CIFAR10C, get_data_loader_CIFAR10
from datasets import load_dataset, Value, ClassLabel

datasets.config.PIL_AVAILABLE = True


# %%[jupynvim:cell-sep]
dataloader = get_data_loader_CIFAR10C(32)
dataset = dataloader.dataset.ds
# %%[jupynvim:cell-sep]
print(dataset.features)
print(dataset.features["image"])
print(dataset.features["label"].names)
print(dataset.features["corruption_name"])
print(dataset.features["corruption_level"])
# %%[jupynvim:cell-sep]
def summarise_feature(dataset, col):
    feature = dataset.features[col]
    series = dataset.to_pandas()[col]
    if isinstance(feature, ClassLabel):
        names = feature.names
        counts = series.value_counts().sort_index()
        print(f"\n{col} ({len(names)} classes)")
        for i, name in enumerate(names):
            print(f"  {name}: {counts.get(i, 0)}")
    else:
        unique = sorted(series.unique())
        counts = series.value_counts().sort_index()
        print(f"\n{col} ({len(unique)} unique values)")
        for val in unique:
            print(f"  {val}: {counts.get(val, 0)}")



for col in ["label", "corruption_name", "corruption_level"]:
    summarise_feature(dataset, col)
# %%[jupynvim:cell-sep]
# Corruption
# %%[jupynvim:cell-sep]

dataloader = get_data_loader_CIFAR10(32, notebook=True)
dataset = dataloader.dataset
# %%[jupynvim:cell-sep]
print(dataset.classes)
# print(dataset.features["corruption_name"])
# print(dataset.features["corruption_level"])
# %%[jupynvim:cell-sep]
def summarise_feature_pytorch(dataset):
      counts = pd.Series(dataset.targets).value_counts().sort_index()
      print(f"\nlabel ({len(dataset.classes)} classes)")
      for i, name in enumerate(dataset.classes):
          print(f"  {name}: {counts.get(i, 0)}")
# %%[jupynvim:cell-sep]
summarise_feature_pytorch(dataset)
# %%[jupynvim:cell-sep]
# Okay so do corruption pipeline
# %%[jupynvim:cell-sep]


# Generate every single corruption -> Save as a paraquet file 
# Then aggregate into a single file 

# That's what the data_loader will handle
