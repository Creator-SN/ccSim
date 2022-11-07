import os
import re
import json
import torch
import random
import thulac
from tqdm import tqdm
from torch.utils.data import Dataset


class SASPromptDataset(Dataset):

    def __init__(self, tokenizer, file_name, padding_length=256, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.thu = thulac.thulac()
        self.cache_path = './tmp/prompt{}'.format(hash(file_name))
        self.comma_re = re.compile(r'[，；、：]')
        if os.path.isfile(self.cache_path):
            print('Resume Dataset Cache...')
            with open(self.cache_path) as f:
                self.compute_list = json.loads(f.read())
        else:
            self.compute_list = self.load_train(file_name)
            if not os.path.isdir('./tmp'):
                os.makedirs('./tmp')
            with open(self.cache_path, 'w') as f:
                f.write(json.dumps(self.compute_list, ensure_ascii=False))
        if shuffle:
            random.shuffle(self.compute_list)

    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            self.ori_list = f.read().split('\n')
        if self.ori_list[len(self.ori_list) - 1] == '':
            self.ori_list = self.ori_list[:len(self.ori_list) - 1]
        word_map = []
        final_list = []
        for line in tqdm(self.ori_list):
            line = line.split('\t')
            gold_ans = line[2]
            words = self.thu.cut(gold_ans)
            for word in words:
                if word[1] == 'n' and word[0] not in word_map:
                    word_map.append(word[0])
            segments = self.comma_re.split(gold_ans)
            index_arr = [idx for idx in range(len(segments))]
            # 抽取子句组合
            for i in range(len(segments)):
                random_index = random.sample(index_arr, i + 1)
                random_index.sort()
                sample_segements = [segments[idx] for idx in random_index]
                concat_sentence = ','.join(sample_segements)
                final_list.append({
                    'text1': gold_ans,
                    'text2': concat_sentence,
                    'prompt_text': '存在遗漏',
                    'label': float((i + 1) / len(segments)),
                    'ori_seg_len': len(segments),
                    'cls_label': 1
                })
            final_list.append({
                'text1': gold_ans,
                'text2': gold_ans,
                'prompt_text': '完美作答',
                'label': 1.0,
                'ori_seg_len': len(segments),
                'cls_label': 0
            })

        threshold = 1
        err_list = []
        for _ in range(threshold):
            for item in tqdm(final_list):
                fake_ans = item['text2']
                concat_sentence = ''
                words = self.thu.cut(fake_ans)
                count = 0
                for word in words:
                    if word[1] == 'n' and random.random() >= 0.85:
                        err_word = random.sample(word_map, 1)[0]
                        if err_word != word[0]:
                            count += 1
                            word[0] = err_word
                    concat_sentence += word[0]
                if count > 0:
                    score = item['label'] - count / item['ori_seg_len']
                    score = score if score > 0 else 0
                    err_list.append({
                        'text1': item['text1'],
                        'text2': concat_sentence,
                        'prompt_text': '存在错误',
                        'label': score,
                        'ori_seg_len': item['ori_seg_len'],
                        'cls_label': 2
                    })

        return final_list + err_list

    def __getitem__(self, idx):
        item = self.compute_list[idx]
        text1 = item['text1']
        text2 = item['text2']
        prompt_text = '[MASK][MASK][MASK][MASK]'
        prompt_label = item['prompt_text']
        CLS = self.tokenizer('[CLS]')['input_ids']
        SEP = self.tokenizer('[SEP]')['input_ids']
        text1_T = self.tokenizer(text1, add_special_tokens=False)
        text2_T = self.tokenizer(text2, add_special_tokens=False)
        prompt_T = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_label_T = self.tokenizer(prompt_label, add_special_tokens=False)
        input_ids = CLS + text1_T['input_ids'] + SEP + \
            text2_T['input_ids'] + SEP + prompt_T['input_ids'] + SEP
        input_ids = input_ids[:self.padding_length]
        input_ids = input_ids + [0] * (self.padding_length - len(input_ids))
        attn_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(text1_T['input_ids']) + 2) + [1] * (
            len(text2_T['input_ids']) + 1) + [1] * (len(prompt_T['input_ids']) + 1)
        token_type_ids = token_type_ids[:self.padding_length]
        token_type_ids = token_type_ids + [1] * (self.padding_length - len(token_type_ids))
        labels = [-100] + [-100 for _ in text1_T['input_ids']] + [-100] + \
            [-100 for _ in text2_T['input_ids']] + \
            [-100] + prompt_label_T['input_ids'] + SEP
        labels = labels[:self.padding_length]
        labels = labels + [-100] * (self.padding_length - len(labels))
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attn_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.compute_list)
