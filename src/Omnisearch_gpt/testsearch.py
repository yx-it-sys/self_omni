from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from collections import defaultdict
import json
import random 

# 验证加载
loaded_data = load_from_disk("D:/gitcode/projects/toy_dataset")
print(f"最终 toy 数据集大小: {len(loaded_data)}")
print(loaded_data[0])