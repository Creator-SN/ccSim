# %%
import json

with open('./s_train_zh.csv') as f:
    ori_list = f.readlines()

with open('./qa_zh.csv') as f:
    qa_list = f.readlines()

if ori_list[-1] == '':
    ori_list = ori_list[:-1]

if qa_list[-1] == '':
    qa_list = qa_list[:-1]

result = []

for idx, line in enumerate(ori_list):
    line = line.strip()
    line = line.split('\t')
    id = line[0]
    question = qa_list[int(id) - 1].split('\t')[1]
    std_answer = qa_list[int(id) - 1].split('\t')[2]
    result.append({
        'id': line[0],
        'question': question,
        'text1': std_answer.strip(),
        'text2': line[3],
        'label': float(line[1]) / 5,
        'category': line[1]
    })

with open('./s_train_zh.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False)

# %%
