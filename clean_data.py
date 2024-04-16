import json
import pandas as pd

df = pd.read_json("train.json")

print(df.shape)
df = df[df['ner'].apply(lambda x: len(x) > 0)]


print(df.shape)
train_dict = df.to_dict('records')
df = pd.read_json("test.json")
print(df.shape)
df = df[df['ner'].apply(lambda x: len(x) > 0)]

print(df.shape)
test_dict = df.to_dict('records')

with open('final_train.json', 'w') as json_file:
    json.dump(train_dict, json_file)
with open('final_test.json', 'w') as json_file:
    json.dump(test_dict, json_file)