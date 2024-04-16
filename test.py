import json
from gliner import GLiNER

model = GLiNER.from_pretrained("finetuned-gliner_large-v2/logs/finetuned_16")

eval_path = "final_test.json"

with open(eval_path, "r") as f:
    eval_data = json.load(f)

results, f1 = model.evaluate(eval_data, flat_ner=True, threshold=0.5, batch_size=5,
                entity_types=['B-EMAIL','B-ID_NUM','B-NAME_STUDENT','B-PHONE_NUM',
            'B-STREET_ADDRESS','B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
            'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS','I-URL_PERSONAL']) #,'O'])

print(results)

