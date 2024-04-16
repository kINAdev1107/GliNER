import json

train_path = 'train_data.json'
test_path = "test_data.json"

with open(train_path, "r") as f:
    data = json.load(f)

with open(test_path, "r") as f:
    data_test = json.load(f)


def group_entities(lst1):
    values = ['B-EMAIL',
        'B-ID_NUM',
        'B-NAME_STUDENT',
        'B-PHONE_NUM',
        'B-STREET_ADDRESS',
        'B-URL_PERSONAL',
        'B-USERNAME',
        'I-ID_NUM',
        'I-NAME_STUDENT',
        'I-PHONE_NUM',
        'I-STREET_ADDRESS',
        'I-URL_PERSONAL',]

    results = []
    indexes = []
    for value in values:
        result = find_indexes(lst1, value)
        if len(result) > 0:
            if len(result) == 1:
                result.append(result[0])
            result.append(value)
            if len(result)>0:
                results.append(result)
            # print(results)
    return results


for idx, val in enumerate(data):
    value = val['ner']
    data[idx]['ner'] = [[idx, idx, string] for idx, string in enumerate(value) if string != "O"]

with open('train.json', 'w') as json_file:
    json.dump(data, json_file)


for idx, val in enumerate(data_test):
    value = val['ner']
    data_test[idx]['ner'] = [[idx, idx, string] for idx, string in enumerate(value) if string != "O"]

with open('test.json', 'w') as json_file:
    json.dump(data_test, json_file)