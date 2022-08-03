# %%
import random
with open('./sentences-goldstandard.csv', encoding='utf-8') as f:
    ori_list = f.read().split('\n')

if ori_list[-1] == '':
    ori_list = ori_list[:-1]

random.shuffle(ori_list)
train_list = ori_list[:int(len(ori_list) / 10 * 7)]
eval_list = ori_list[int(len(ori_list) / 10 * 7):]

with open('./s_train_zh.csv', encoding='utf-8', mode='a+') as f:
    for line in train_list:
        f.write('{}\n'.format(line))

with open('./s_eval_zh.csv', encoding='utf-8', mode='a+') as f:
    for line in eval_list:
        f.write('{}\n'.format(line))

# %%
