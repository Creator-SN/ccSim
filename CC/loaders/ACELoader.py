# %%
import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class ACEDataset(Dataset):
    def __init__(self, tokenizer, file_name, padding_length=512, max_word_size=6, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.file_name = file_name
        self.ori_json = self.load_train(file_name)
        self.max_word_size = max_word_size
        self.random_idx_list = [idx for idx in range(len(self.ori_json))]
        if shuffle:
            random.shuffle(self.random_idx_list)

    def load_train(self, file_name):
        if file_name.endswith('json'):
            with open(file_name, encoding='utf-8') as f:
                ori_json = f.read()
            ori_json = json.loads(ori_json)
        else:
            with open(file_name, encoding='utf-8') as f:
                ori_json = f.readlines()
            ori_json = [json.loads(item) for item in ori_json]
        return ori_json

    def __getitem__(self, idx):
        idx = self.random_idx_list[idx]
        item = self.ori_json[idx]
        text1 = item['text1']
        text2 = item['text2']
        labels = int(item['label'])
        text1_words = item['text1_words']
        text2_words = item['text2_words']
        text1_words_text = ''
        text2_words_text = ''

        for i, item in enumerate(text1_words):
            if i >= self.max_word_size:
                break
            item_prompt = f'&{item["entity"]}是{item["type"]}'
            text1_words_text += item_prompt

        for i, item in enumerate(text2_words):
            if i >= self.max_word_size:
                break
            item_prompt = f'&{item["entity"]}是{item["type"]}'
            text2_words_text += item_prompt


        T = self.tokenizer(text1, text2, add_special_tokens=True,
                               max_length=self.padding_length, padding='max_length', truncation=True, return_offsets_mapping=True)
        T_e = self.tokenizer(text1_words_text, text2_words_text, add_special_tokens=True,
                               max_length=self.padding_length, padding='max_length', truncation=True, return_offsets_mapping=True)

        return {
            't_ids': torch.tensor(T['input_ids']),
            't_masks': torch.tensor(T['attention_mask']),
            't_types': torch.tensor(T['token_type_ids']),
            'te_ids': torch.tensor(T_e['input_ids']),
            'te_masks': torch.tensor(T_e['attention_mask']),
            'te_types': torch.tensor(T_e['token_type_ids']),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.ori_json)
