import json
import pandas as pd

dataset = pd.read_csv("train-preprocessed.csv")
signs = dataset['sign'].unique()

json_data = {}
for index, value in enumerate(signs):
  json_data[value] = index

with open("sign2label.json", "w") as file:
  json.dumps(json_data, file, indent=4)

print("sign2label.json file created...")
