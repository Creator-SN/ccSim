# %%
import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class ACSTSDataset(Dataset):
    def __init__(self, tokenizer, file_name, lexicon_path, padding_length=128, max_word_len=30, model_type='interactive', shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.model_type = model_type
        self.file_name = file_name
        self.ori_json = self.load_train(file_name)
        self.lexicon_list, self.matched_word_dict = self.load_lexicon(
            lexicon_path)
        self.max_word_len = max_word_len
        self.random_idx_list = [idx for idx in range(len(self.ori_json))]
        if shuffle:
            random.shuffle(self.random_idx_list)
        self.compute_match_word_list()

    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_json = f.read()
        ori_json = json.loads(ori_json)
        return ori_json

    def load_lexicon(self, lexicon_path):
        with open(lexicon_path, mode='r') as f:
            ori_list = f.read().split('\n')

        if ori_list[-1] == '':
            ori_list = ori_list[:-1]

        result = [item.split(',')[0] for item in ori_list]
        matched_word_dict = {}
        for idx, word in enumerate(result):
            matched_word_dict[word] = idx + 1
        matched_word_dict['[MASK]'] = 0

        return result, matched_word_dict

    def match_word(self, seqence, max_len=8):
        match_list = []
        mask_list = []
        seq_matches = []
        seq_mask = []
        count = 0
        for i in range(len(seqence)):
            matches = []
            for j in range(i + 1, len(seqence)):
                word = seqence[i:j]
                if word in self.matched_word_dict:
                    matches.append(self.matched_word_dict[word])
                    count += 1

                    if word not in seq_matches:
                        seq_matches.append(self.matched_word_dict[word])
            matches = matches[:max_len]
            remain = max_len - len(matches)
            for i in range(remain):
                matches.append(0)
            mask = [1 if m != 0 else 0 for m in matches]
            match_list.append(matches)
            mask_list.append(mask)
        
        seq_matches = seq_matches[:max_len]
        remain = max_len - len(seq_matches)
        for i in range(remain):
            seq_matches.append(0)
        seq_mask = [1 if m != 0 else 0 for m in seq_matches]
        return match_list, mask_list, seq_matches, seq_mask, count
    
    def compute_match_word_list(self):
        self.matched_word_ids_list = []
        self.matched_word_mask_list = []
        self.sequence_word_ids_list = []
        self.sequence_word_mask_list = []
        cache_path = './tmp/{}_matched_word_{}'.format(self.file_name.replace('/', '___'), self.padding_length)
        if os.path.exists(cache_path):
            with open(os.path.join(cache_path, 'ids'), 'rb') as f:
                self.matched_word_ids_list = pickle.load(f)
            with open(os.path.join(cache_path, 'mask'), 'rb') as f:
                self.matched_word_mask_list = pickle.load(f)
            with open(os.path.join(cache_path, 'seq_ids'), 'rb') as f:
                self.sequence_word_ids_list = pickle.load(f)
            with open(os.path.join(cache_path, 'seq_mask'), 'rb') as f:
                self.sequence_word_mask_list = pickle.load(f)
            return True
        
        count = []
        for idx, item in tqdm(enumerate(self.ori_json)):
            s1 = item['text1']
            s2 = item['text2']
            s1 = s1[:self.padding_length // 2]
            s2 = s2[:self.padding_length // 2]
            if self.model_type == 'interactive':
                T = self.tokenizer(s1, s2, add_special_tokens=True,
                                max_length=self.padding_length, padding='max_length', truncation=True)
                input_ids = torch.tensor(T['input_ids'])

                mw_s1, mm_s1, seqw_s1, seqm_s1, count1 = self.match_word(s1, max_len=self.max_word_len)
                mw_s2, mm_s2, seqw_s2, seqm_s2, count2 = self.match_word(s2, max_len=self.max_word_len)
                count.append(count1 + count2)
                mask_token_words = [[0 for _ in range(self.max_word_len)]]
                mask_token_mask = [[0 for _ in range(self.max_word_len)]]

                matched_word_ids = mask_token_words + mw_s1 + mask_token_words + mw_s2
                matched_word_mask = mask_token_mask + mm_s1 + mask_token_mask + mm_s2
                sequence_word_ids = [seqw_s1] + [seqw_s2]
                sequence_word_mask = [seqm_s1] + [seqm_s2]
                remain = len(input_ids) - len(matched_word_ids)
                for _ in range(remain):
                    matched_word_ids += mask_token_words
                    matched_word_mask += mask_token_mask

                self.matched_word_ids_list.append(matched_word_ids)
                self.matched_word_mask_list.append(matched_word_mask)
            
            elif self.model_type == 'siamese':
                left_length = self.padding_length // 2
                if left_length < self.padding_length / 2:
                    left_length += 1
                right_length = self.padding_length - left_length
                s1 = s1[:left_length]
                s2 = s2[:right_length]
                T1 = self.tokenizer(s1, add_special_tokens=True,
                                    max_length=left_length, padding='max_length', truncation=True)
                T2 = self.tokenizer(s2, add_special_tokens=True,
                                    max_length=right_length, padding='max_length', truncation=True)
                ss1 = torch.tensor(T1['input_ids'])
                ss2 = torch.tensor(T2['input_ids'])

                mw_s1, mm_s1, seqw_s1, seqm_s1, count1 = self.match_word(s1, max_len=self.max_word_len)
                mw_s2, mm_s2, seqw_s2, seqm_s2, count2 = self.match_word(s2, max_len=self.max_word_len)
                count.append(count1 + count2)
                mask_token_words = [[0 for _ in range(self.max_word_len)]]
                mask_token_mask = [[0 for _ in range(self.max_word_len)]]
                
                matched_word_ids1 = mask_token_words + mw_s1
                matched_word_mask1 = mask_token_mask + mm_s1
                remain = len(ss1) - len(matched_word_ids1)
                for _ in range(remain):
                    matched_word_ids1 += mask_token_words
                    matched_word_mask1 += mask_token_mask
                # 当句子长度等于padding_length时，matched_word_ids1的长度会超过padding_length
                matched_word_ids1 = matched_word_ids1[:left_length]
                matched_word_mask1 = matched_word_mask1[:left_length]
                
                matched_word_ids2 = mask_token_words + mw_s2
                matched_word_mask2 = mask_token_mask + mm_s2
                remain = len(ss2) - len(matched_word_ids2)
                for _ in range(remain):
                    matched_word_ids2 += mask_token_words
                    matched_word_mask2 += mask_token_mask
                matched_word_ids2 = matched_word_ids2[:right_length]
                matched_word_mask2 = matched_word_mask2[:right_length]
                
                matched_word_ids = matched_word_ids1 + matched_word_ids2
                matched_word_mask = matched_word_mask1 + matched_word_mask2
                sequence_word_ids = [seqw_s1] + [seqw_s2]
                sequence_word_mask = [seqm_s1] + [seqm_s2]
                
                self.matched_word_ids_list.append(matched_word_ids)
                self.matched_word_mask_list.append(matched_word_mask)
                self.sequence_word_ids_list.append(sequence_word_ids)
                self.sequence_word_mask_list.append(sequence_word_mask)
        
        print('Collected {} matched words, average {}, max {}, min {}\n'.format(np.sum(count), np.mean(count), np.max(count), np.min(count)))

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        with open(os.path.join(cache_path, 'ids'), 'wb') as f:
            pickle.dump(self.matched_word_ids_list, f, 2)
        with open(os.path.join(cache_path, 'mask'), 'wb') as f:
            pickle.dump(self.matched_word_mask_list, f, 2)
        with open(os.path.join(cache_path, 'seq_ids'), 'wb') as f:
            pickle.dump(self.sequence_word_ids_list, f, 2)
        with open(os.path.join(cache_path, 'seq_mask'), 'wb') as f:
            pickle.dump(self.sequence_word_mask_list, f, 2)

    def __getitem__(self, idx):
        idx = self.random_idx_list[idx]
        item = self.ori_json[idx]
        s1 = item['text1']
        s2 = item['text2']
        s1 = s1[:self.padding_length // 2]
        s2 = s2[:self.padding_length // 2]
        labels = int(item['label'])
        if self.model_type == 'interactive':
            T = self.tokenizer(s1, s2, add_special_tokens=True,
                               max_length=self.padding_length, padding='max_length', truncation=True)
            input_ids = torch.tensor(T['input_ids'])
            attn_mask = torch.tensor(T['attention_mask'])
            token_type_ids = torch.tensor(T['token_type_ids'])

            matched_word_ids = self.matched_word_ids_list[idx]
            matched_word_mask = self.matched_word_mask_list[idx]
            sequence_word_ids = self.sequence_word_ids_list[idx]
            sequence_word_mask = self.sequence_word_mask_list[idx]

            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'token_type_ids': token_type_ids,
                'matched_word_ids': torch.tensor(matched_word_ids),
                'matched_word_mask': torch.tensor(matched_word_mask),
                'sequence_word_ids': torch.tensor(sequence_word_ids),
                'sequence_word_mask': torch.tensor(sequence_word_mask),
                'labels': torch.tensor(labels)
            }
        
        elif self.model_type == 'siamese':
            left_length = self.padding_length // 2
            if left_length < self.padding_length / 2:
                left_length += 1
            right_length = self.padding_length - left_length
            s1 = s1[:left_length]
            s2 = s2[:right_length]
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
            
            matched_word_ids = self.matched_word_ids_list[idx]
            matched_word_mask = self.matched_word_mask_list[idx]
            sequence_word_ids = self.sequence_word_ids_list[idx]
            sequence_word_mask = self.sequence_word_mask_list[idx]
            
            return {
                'input_ids': torch.cat([ss1, ss2]),
                'attention_mask': torch.cat([mask1, mask2]),
                'token_type_ids': torch.cat([tid1, tid2]),
                'matched_word_ids': torch.tensor(matched_word_ids),
                'matched_word_mask': torch.tensor(matched_word_mask),
                'sequence_word_ids': torch.tensor(sequence_word_ids),
                'sequence_word_mask': torch.tensor(sequence_word_mask),
                'labels': torch.tensor(labels)
            }

    def __len__(self):
        return len(self.ori_json)


class WordTokenizer():

    def __init__(self, vocab_file):
        with open(vocab_file, mode='r') as f:
            ori_list = f.read().split('\n')

        if ori_list[:-1] == '':
            ori_list = ori_list[:-1]

        self.vocab_list = ori_list

        self.token_2_id = {}
        self.id_2_token = {}
        for id, token in enumerate(self.vocab_list):
            self.token_2_id[token] = id
            self.id_2_token[id] = token

    def token_to_id(self, token):
        if token not in self.token_2_id:
            return self.token_2_id['[UNK]']
        return self.token_2_id[token]

    def id_to_token(self, id):
        id = int(id)
        return self.id_2_token[id]

    def convert_tokens_to_ids(self, sequence):
        ids = []
        for token in sequence:
            ids.append(self.token_to_id(token))
        return ids

    def __call__(self, sequence1, sequence2=None, add_special_tokens=True, max_length=128, padding='max_length', truncation=True):
        return self.encode(sequence1, sequence2, add_special_tokens,
                    max_length, padding, truncation)

    def encode(self, sequence1, sequence2=None, add_special_tokens=True, max_length=128, padding='max_length', truncation=True):
        t1 = self.convert_tokens_to_ids(sequence1)
        if sequence2 is not None:
            t2 = self.convert_tokens_to_ids(sequence2)
            if add_special_tokens:
                max_half_len = max_length // 2 - 1
            else:
                max_half_len = max_length // 2
            if padding == 'max_length':
                remain = max_half_len - len(t1)
                for _ in range(remain):
                    t1.append(self.token_to_id('[PAD]'))

                remain = max_half_len - len(t2)
                for _ in range(remain):
                    t2.append(self.token_to_id('[PAD]'))
            if truncation:
                t1 = t1[:max_half_len]
                t2 = t2[:max_half_len]

            if add_special_tokens:
                input_ids = [self.token_to_id(
                    '[CLS]')] + t1 + [self.token_to_id('[SEP]')] + t2
                token_type_ids = [0] + [0 for _ in t1] + [1] + [1 for _ in t2]
            else:
                input_ids = t1 + t2
                token_type_ids = [0 for _ in input_ids]
        else:
            if add_special_tokens:
                max_half_len = max_length - 1
            else:
                max_half_len = max_length

            if padding == 'max_length':
                remain = max_half_len - len(t1)
                for idx in range(remain):
                    if idx == 0:
                        t1.append(self.token_to_id('[SEP]'))
                    else:
                        t1.append(self.token_to_id('[PAD]'))

            if truncation:
                t1 = t1[:max_half_len]

            if add_special_tokens:
                input_ids = [self.token_to_id(
                    '[CLS]')] + t1
            else:
                input_ids = t1

            token_type_ids = [0 for _ in input_ids]

        attention_mask = [1 if token != self.token_to_id(
            '[PAD]') else 0 for token in input_ids]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

    def decode(self, ids, skip_special_tokens=False):
        return ''.join([self.id_to_token(id) for id in ids])
