# %%
with open('./SAG/ori_train', encoding='utf-8') as f:
    ori_list = f.read().split('\n')

id = 0
result = ''
for line in ori_list:
    line = line.strip().split('\t')
    if len(line[0]) > 0 and line[0][0] == '#':
        id += 1
    elif line[0] == '':
        continue
    else:
        try:
            result += '{}\t{}\t{}\t{}\n'.format(id, line[0], line[1], line[2])
        except:
            print(line, line[0])

with open('./SAG/final_train.csv', encoding='utf-8', mode='w+') as f:
    f.write(result)
# %%
