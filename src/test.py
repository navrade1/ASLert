import json
from pathlib import Path

def process(file_info: dict):
    ...

json_file = Path('../data/labels.json')

with open(json_file) as f:
    labels = json.load(f)

for file_info in labels:
    process(file_info)