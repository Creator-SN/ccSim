# %%
import os
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

class CLDataset(Dataset):
    def __init__(self, tokenizer, file_name, padding_length=128, model_type='interactive', shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.model_type = model_type
        self.ori_data, self.compute_data = self.load_train(file_name)
        if shuffle:
            random.shuffle(self.compute_data)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_data = f.read().split('\n')
        if ori_data[-1] == '':
            ori_data = ori_data[:-1]
        result = []
        symbols = ['。', '，', '、', ')']
        for line in ori_data:
            while len(line) > self.padding_length:
                tmp = line[:self.padding_length]
                rindex_list = []
                for symbol in symbols:
                    try:
                        rindex_list.append(tmp.rindex(symbol))
                    except:
                        pass
                sorted(rindex_list, reverse=True)
                if len(rindex_list) > 0:
                    last_index = rindex_list[0] + 1
                else:
                    last_index = self.padding_length
                result.append(line[:last_index])
                line = line[last_index:]
            result.append(line)
        return ori_data, result
    
    def __getitem__(self, idx):
        s1 = self.compute_data[idx]
        s2 = s1
        if self.model_type == 'interactive':
            T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
            input_ids = torch.tensor(T['input_ids'])
            attn_mask = torch.tensor(T['attention_mask'])
            token_type_ids = torch.tensor(T['token_type_ids'])
            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'token_type_ids': token_type_ids,
                'labels': torch.tensor(1)
            }
        elif self.model_type == 'siamese':
            left_length = self.padding_length // 2
            if left_length < self.padding_length / 2:
                left_length += 1
            right_length = self.padding_length - left_length
            T1 = self.tokenizer(s1, add_special_tokens=True, max_length=left_length, padding='max_length', truncation=True)
            T2 = self.tokenizer(s2, add_special_tokens=True, max_length=right_length, padding='max_length', truncation=True)
            ss1 = torch.tensor(T1['input_ids'])
            mask1 = torch.tensor(T1['attention_mask'])
            tid1 = torch.tensor(T1['token_type_ids'])
            ss2 = torch.tensor(T2['input_ids'])
            mask2 = torch.tensor(T2['attention_mask'])
            tid2 = torch.ones(ss2.shape).long()
            return {
                'input_ids': torch.cat([ss1, ss2]),
                'attention_mask': torch.cat([mask1, mask2]),
                'token_type_ids': torch.cat([tid1, tid2]),
                'labels': torch.tensor(1)
            }
    
    def __len__(self):
        return len(self.compute_data)
