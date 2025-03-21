# %%
import json

with open('test_atec.csv') as f:
    ori_data = f.read().split('\n')

if ori_data[-1] == '':
    ori_data = ori_data[:-1]

result = []

for idx, line in enumerate(ori_data):
    if line == '':
        continue
    line = line.split('\t')
    item = {
        'id': idx,
        'text1': line[1],
        'text2': line[2],
        'label': line[3]
    }
    result.append(item)

with open('test.json', 'w') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

# %%
with open('./THUOCL_caijing.txt') as f:
    ori_data = f.read().split('\n')

if ori_data[-1] == '':
    ori_data = ori_data[:-1]

result = []

for idx, line in enumerate(ori_data):
    line = line.split(' 	 ')
    entity = line[0]
    result.append(entity)

with open('ner', 'w') as f:
    for item in result:
        f.write(item + '\n')


# %%
import os
import json
import random
import shutil

samples = [250, 500, 1000, 2000]
sorted(samples)

with open('train.json') as f:
    ori_data = json.load(f)
max_sample_result = random.sample(ori_data, samples[-1])

for sample in samples:
    sample_result = max_sample_result[:sample]
    if not os.path.exists('../fewshot/ATEC_{}'.format(sample)):
        os.makedirs('../fewshot/ATEC_{}'.format(sample))

    with open('../fewshot/ATEC_{}/train.json'.format(sample), 'w') as f:
        json.dump(sample_result, f, indent=4, ensure_ascii=False)

    if os.path.exists('dev.json'):
        shutil.copyfile('dev.json', '../fewshot/ATEC_{}/dev.json'.format(sample))

    if os.path.exists('test.json'):
        shutil.copyfile('test.json', '../fewshot/ATEC_{}/test.json'.format(sample))

# %%
