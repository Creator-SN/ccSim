# %%
import os
import json
import random

with open('./s_train_zh.json', encoding='utf-8') as f:
    ori_list = f.read()
ori_list = json.loads(ori_list)

sample_list = random.sample(ori_list, 100)

if not os.path.isdir('./few_shot'):
    os.makedirs('./few_shot')

with open('./few_shot/s_train_zh.json', encoding='utf-8', mode='w+') as f:
    f.write(json.dumps(sample_list, ensure_ascii=False))

# %%
