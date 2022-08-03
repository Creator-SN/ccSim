# %%
with open('./sentences-goldstandard.csv', encoding='utf-8') as f:
    ori_list = f.read().split('\n')

with open('./gold_ans.csv', encoding='utf-8') as f:
    gold_list = f.read().split('\n')
    if gold_list[-1] == '':
        gold_list = gold_list[:-1]
    gold_dict = {}
for idx, line in enumerate(gold_list):
    line = line.split('\t')
    id, ans = line[0], line[1]
    gold_dict[id] = [idx, ans]
    
weakly_1 = ''
weakly_2 = ''
for line in ori_list:
    if line.strip() == '':
        continue
    if gold_dict[line.split('\t')[6]][0] < 17:
        weakly_1 += line + '\n'
    else:
        weakly_2 += line + '\n'

with open('./weakly/s_w1_zh.csv', encoding='utf-8', mode='w+') as f:
    f.write(weakly_1)

with open('./weakly/s_w2_zh.csv', encoding='utf-8', mode='w+') as f:
    f.write(weakly_2)

# %%
