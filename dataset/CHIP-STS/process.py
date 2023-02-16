# %%
import os
import json
import random
import shutil

samples = [250, 500, 1000, 2000]
sorted(samples)

with open('CHIP-STS_train.json') as f:
    ori_data = json.load(f)
max_sample_result = random.sample(ori_data, samples[-1])

for sample in samples:
    sample_result = max_sample_result[:sample]

    if not os.path.exists('../fewshot/CHIP-STS_{}'.format(sample)):
        os.makedirs('../fewshot/CHIP-STS_{}'.format(sample))

    with open('../fewshot/CHIP-STS_{}/CHIP-STS_train.json'.format(sample), 'w') as f:
        json.dump(sample_result, f, indent=4, ensure_ascii=False)

    if os.path.exists('CHIP-STS_dev.json'):
        shutil.copyfile('CHIP-STS_dev.json', '../fewshot/CHIP-STS_{}/dev.json'.format(sample))

    if os.path.exists('CHIP-STS_test.json'):
        shutil.copyfile('CHIP-STS_test.json', '../fewshot/CHIP-STS_{}/test.json'.format(sample))

# %%
