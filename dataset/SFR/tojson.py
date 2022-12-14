# %%
import json

with open('./s_eval_zh.csv') as f:
    ori_list = f.readlines()

if ori_list[-1] == '':
    ori_list = ori_list[:-1]

result = []

for idx, line in enumerate(ori_list):
    line = line.strip()
    line = line.split('\t')
    result.append({
        'id': line[0],
        'text1': line[1],
        'text2': line[7],
        'label': line[3],
        'category': line[2]
    })

with open('./s_eval_zh.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False)

# %%
