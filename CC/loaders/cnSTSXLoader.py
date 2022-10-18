# %%
import os
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class CNSTSXDataset(Dataset):
    def __init__(self, tokenizer, file_name, vocab_file, padding_length=128, model_type='interactive', vocab_size=10, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.model_type = model_type
        self.ori_json = self.load_train(file_name)
        self.dictionary_list, self.word_to_idx = self.read_dictionary(vocab_file)
        self.vocab_list = self.compute_data_vocab(vocab_size)
        self.data_index = [i for i in range(len(self.ori_json))]
        if shuffle:
            random.shuffle(self.data_index)

    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_json = f.read()
        ori_json = json.loads(ori_json)
        return ori_json

    def read_dictionary(self, filename):
        """Reads a dictionary file and returns a list of words.
        filename: string
        returns: list of strings
        """
        with open(filename, encoding='utf-8') as f:
            ori_list = f.read().splitlines()
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        result = []
        result_dict = {}
        for idx, word in enumerate(ori_list):
            result.append(word.strip())
            result_dict[word.strip()] = idx + 1
        return result, result_dict

    def get_words(self, word_list, sentence):
        """Returns a list of words from a sentence.
        word_list: list of strings
        sentence: string
        returns: list of strings
        """
        words = []
        for word in word_list:
            if word in sentence:
                words.append(self.word_to_idx[word])
        return words

    def compute_data_vocab(self, max_vocab):
        vocab_list = []
        for item in tqdm(self.ori_json):
            s1 = item['text1']
            s2 = item['text2']
            words_s1 = self.get_words(self.dictionary_list, s1)
            words_s2 = self.get_words(self.dictionary_list, s2)
            words_s1 = words_s1[:max_vocab]
            words_s2 = words_s2[:max_vocab]
            if len(words_s1) < max_vocab:
                words_s1 += [0] * (max_vocab - len(words_s1))
            if len(words_s2) < max_vocab:
                words_s2 += [0] * (max_vocab - len(words_s2))
            vocab_list.append({
                'text1_words': words_s1,
                'text2_words': words_s2
            })
        return vocab_list

    def __getitem__(self, idx):
        item = self.ori_json[self.data_index[idx]]
        words = self.vocab_list[self.data_index[idx]]
        s1 = item['text1']
        s2 = item['text2']
        s1_words = words['text1_words']
        s2_words = words['text2_words']
        words_input_ids_1 = torch.tensor(s1_words)
        words_attention_mask_1 = words_input_ids_1.gt(0)
        words_input_ids_2 = torch.tensor(s2_words)
        words_attention_mask_2 = words_input_ids_2.gt(0)
        labels = int(item['label'])
        if self.model_type == 'interactive':
            T = self.tokenizer(s1, s2, add_special_tokens=True,
                               max_length=self.padding_length, padding='max_length', truncation=True)
            input_ids = torch.tensor(T['input_ids'])
            attn_mask = torch.tensor(T['attention_mask'])
            token_type_ids = torch.tensor(T['token_type_ids'])
            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'token_type_ids': token_type_ids,
                'words_input_ids_1': words_input_ids_1,
                'words_attention_mask_1': words_attention_mask_1,
                'words_input_ids_2': words_input_ids_2,
                'words_attention_mask_2': words_attention_mask_2,
                'labels': torch.tensor(labels)
            }
        elif self.model_type == 'siamese':
            left_length = self.padding_length // 2
            if left_length < self.padding_length / 2:
                left_length += 1
            right_length = self.padding_length - left_length
            T1 = self.tokenizer(s1, add_special_tokens=True,
                                max_length=left_length, padding='max_length', truncation=True)
            T2 = self.tokenizer(s2, add_special_tokens=True,
                                max_length=right_length, padding='max_length', truncation=True)
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
                'words_input_ids_1': words_input_ids_1,
                'words_attention_mask_1': words_attention_mask_1,
                'words_input_ids_2': words_input_ids_2,
                'words_attention_mask_2': words_attention_mask_2,
                'labels': torch.tensor(labels)
            }

    def __len__(self):
        return len(self.ori_json)
