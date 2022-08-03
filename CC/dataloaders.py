# %%
import os
import torch
import random
import thulac
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class STSDataset(Dataset):

    def __init__(self, tokenizer, file_name, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(file_name)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        return ori_list
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[5], line[6], line[4]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label) / 5)
    
    def __len__(self):
        return len(self.ori_list)

class STSDataset_Bert_ESIM(STSDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[5], line[6], line[4]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return ss1, mask1.eq(0), tid1, ss2, mask2.eq(0), tid2, torch.tensor(float(label) / 5)

class STSDataset_Cosine(STSDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[5], line[6], line[4]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label) / 5)

class SAGDataset(Dataset):

    def __init__(self, tokenizer, dir_name, padding_length=128, mode='train', shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.qa_list = self.load_qa(os.path.join(dir_name, 'qa_zh.csv'))
        self.ori_list = self.load_train(os.path.join(dir_name, 's_{}_zh.csv'.format(mode)))
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_qa(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        result_list = {}
        for line in ori_list:
            line = line.split('\t')
            result_list[line[0]] = line
        return result_list
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        return ori_list
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        std_id, label, s1 = line[0], line[1], line[3]
        line = self.qa_list[std_id]
        s2 = line[2]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label) / 5)
    
    def __len__(self):
        return len(self.ori_list)

class SAGCosine(SAGDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        std_id, label, s1 = line[0], line[1], line[3]
        line = self.qa_list[std_id]
        s2 = line[2]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label) / 5)


class GenerationSTS(STSDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[5], line[6], line[4]
        T = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T_tags = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)

        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        tags = torch.tensor(T_tags['input_ids'])

        ratio = 1 - float(label) / 5
        index_arr = [i for i in range(len(input_ids))]
        index_arr = index_arr[1:]
        random.shuffle(index_arr)
        index_arr = index_arr[:int(len(index_arr) * ratio)]
        masked_arr = index_arr[:int(len(index_arr) * 0.8)]
        err_arr = index_arr[int(len(index_arr) * 0.8):int(len(index_arr) * 0.9)]

        for idx in masked_arr:
            input_ids[idx] = 103
            tags[idx] = -100
        for idx in err_arr:
            input_ids[idx] = int(random.random() * 21120)
        
        return input_ids, attn_mask, token_type_ids, tags

class AliDataset(STSDataset):
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[2], line[3]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return ss1, mask1.eq(0), tid1, ss2, mask2.eq(0), tid2, torch.tensor(float(label))

class STSWeaklyZH(Dataset):

    def __init__(self, tokenizer, file_name, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.thu = thulac.thulac()
        self.cache_path = './__datasetcache__/sts.x'
        if os.path.isfile(self.cache_path):
            print('Resume Dataset Cache...')
            with open(self.cache_path, 'rb') as f:
                self.ori_list = pickle.load(f)
        else:
            self.ori_list = self.load_train(file_name)
            if not os.path.isdir('./__datasetcache__'):
                os.makedirs('./__datasetcache__')
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.ori_list, f, 2)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        word_map = []
        final_list = []
        for line in tqdm(ori_list):
            line = line.split('\t')
            gold_ans = line[5]
            words = self.thu.cut(gold_ans)
            for word in words:
                if word[1] == 'n':
                    word_map.append(word[0])
            segments = gold_ans.split('，')
            index_arr = [idx for idx in range(len(segments))]
            for i in range(len(segments)):
                random_index = random.sample(index_arr, i + 1)
                random_index.sort()
                concat_sentence = ''
                for idx in random_index:
                    concat_sentence += ',{}'.format(segments[idx]) if concat_sentence == '' else segments[idx]
                final_list.append([concat_sentence, gold_ans, float(i / len(segments)), len(segments)])
        
        threshold = 1
        err_list = []
        for epoch in range(threshold):
            for item in tqdm(final_list):
                fake_ans = item[0]
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
                    score = item[2] - count / item[3]
                    score = score if score > 0 else 0
                    err_list.append([concat_sentence, item[1], score, item[3]])

        return final_list + err_list
    
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label))
    
    def __len__(self):
        return len(self.ori_list)

class STSCosine_Weakly(STSWeaklyZH):

    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SAGWeaklyZH(Dataset):

    def __init__(self, tokenizer, qa_name, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.thu = thulac.thulac()
        self.cache_path = './__datasetcache__/sag.x'
        if os.path.isfile(self.cache_path):
            print('Resume Dataset Cache...')
            with open(self.cache_path, 'rb') as f:
                self.ori_list = pickle.load(f)
        else:
            self.ori_list = self.load_train(qa_name)
            if not os.path.isdir('./__datasetcache__'):
                os.makedirs('./__datasetcache__')
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.ori_list, f, 2)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, qa_name):
        with open(qa_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        word_map = []
        final_list = []
        for line in tqdm(ori_list):
            line = line.split('\t')
            gold_ans = line[2]
            words = self.thu.cut(gold_ans)
            for word in words:
                if word[1] == 'n':
                    word_map.append(word[0])
            segments = gold_ans.split('，')
            index_arr = [idx for idx in range(len(segments))]
            for i in range(len(segments)):
                random_index = random.sample(index_arr, i + 1)
                random_index.sort()
                concat_sentence = ''
                for idx in random_index:
                    concat_sentence += ',{}'.format(segments[idx]) if concat_sentence == '' else segments[idx]
                final_list.append([concat_sentence, gold_ans, float(i / len(segments)), len(segments)])
        
        threshold = 50
        err_list = []
        for epoch in range(threshold):
            for item in tqdm(final_list):
                fake_ans = item[0]
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
                    score = item[2] - count / item[3]
                    score = score if score > 0 else 0
                    err_list.append([concat_sentence, item[1], score, item[3]])

        return final_list + err_list
    
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label))
    
    def __len__(self):
        return len(self.ori_list)

class SAGCosine_Weakly(SAGWeaklyZH):
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SFRDataset(SAGDataset):

    def __init__(self, tokenizer, dir_name, padding_length=128, mode='train', shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(os.path.join(dir_name, 's_{}_zh.csv'.format(mode)))
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        return ori_list
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[7], line[3]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label))
    
    def __len__(self):
        return len(self.ori_list)

class SFRCosine(SFRDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[7], line[3]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SFRWeaklyZH(Dataset):

    def __init__(self, tokenizer, file_name, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.thu = thulac.thulac()
        self.cache_path = './__datasetcache__/sfr.x'
        if os.path.isfile(self.cache_path):
            print('Resume Dataset Cache...')
            with open(self.cache_path, 'rb') as f:
                self.ori_list = pickle.load(f)
        else:
            self.ori_list = self.load_train(file_name)
            if not os.path.isdir('./__datasetcache__'):
                os.makedirs('./__datasetcache__')
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.ori_list, f, 2)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        word_map = []
        final_list = []
        for line in tqdm(ori_list):
            line = line.split('\t')
            gold_ans = line[1]
            words = self.thu.cut(gold_ans)
            for word in words:
                if word[1] == 'n':
                    word_map.append(word[0])
            segments = gold_ans.split('，')
            index_arr = [idx for idx in range(len(segments))]
            for i in range(len(segments)):
                random_index = random.sample(index_arr, i + 1)
                random_index.sort()
                concat_sentence = ''
                for idx in random_index:
                    concat_sentence += ',{}'.format(segments[idx]) if concat_sentence == '' else segments[idx]
                final_list.append([concat_sentence, gold_ans, float(i / len(segments)), len(segments)])
        
        threshold = 10
        err_list = []
        for epoch in range(threshold):
            for item in tqdm(final_list):
                fake_ans = item[0]
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
                    score = item[2] - count / item[3]
                    score = score if score > 0 else 0
                    err_list.append([concat_sentence, item[1], score, item[3]])

        return final_list + err_list
    
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label))
    
    def __len__(self):
        return len(self.ori_list)

class SFRCosine_Weakly(SFRWeaklyZH):
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SASDataset(SAGDataset):

    def __init__(self, tokenizer, dir_name, padding_length=128, mode='train', shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(os.path.join(dir_name, 's_{}_zh.csv'.format(mode)))
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        return ori_list
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[2], line[3]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label))
    
    def __len__(self):
        return len(self.ori_list)

class SASCosine(SASDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[2], line[3]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SASWeaklyZH(Dataset):

    def __init__(self, tokenizer, file_name, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.thu = thulac.thulac()
        self.cache_path = './__datasetcache__/sas.x'
        if os.path.isfile(self.cache_path):
            print('Resume Dataset Cache...')
            with open(self.cache_path, 'rb') as f:
                self.ori_list = pickle.load(f)
        else:
            self.ori_list = self.load_train(file_name)
            if not os.path.isdir('./__datasetcache__'):
                os.makedirs('./__datasetcache__')
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.ori_list, f, 2)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        word_map = []
        final_list = []
        for line in tqdm(ori_list):
            line = line.split('\t')
            gold_ans = line[2]
            words = self.thu.cut(gold_ans)
            for word in words:
                if word[1] == 'n':
                    word_map.append(word[0])
            segments = gold_ans.split('，')
            index_arr = [idx for idx in range(len(segments))]
            for i in range(len(segments)):
                random_index = random.sample(index_arr, i + 1)
                random_index.sort()
                concat_sentence = ''
                for idx in random_index:
                    concat_sentence += ',{}'.format(segments[idx]) if concat_sentence == '' else segments[idx]
                final_list.append([concat_sentence, gold_ans, float(i / len(segments)), len(segments)])
        
        threshold = 10
        err_list = []
        for epoch in range(threshold):
            for item in tqdm(final_list):
                fake_ans = item[0]
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
                    score = item[2] - count / item[3]
                    score = score if score > 0 else 0
                    err_list.append([concat_sentence, item[1], score, item[3]])

        return final_list + err_list
    
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids, torch.tensor(float(label))
    
    def __len__(self):
        return len(self.ori_list)

class SASCosine_Weakly(SASWeaklyZH):
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SAGESIM(SAGDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        std_id, label, s1 = line[0], line[1], line[3]
        line = self.qa_list[std_id]
        s2 = line[2]
        T1 = self.tokenizer(s1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label) / 5)

class SAGESIM_Weakly(SAGWeaklyZH):
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SASESIM(SASDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[2], line[3]
        T1 = self.tokenizer(s1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SASESIM_Weakly(SASWeaklyZH):
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SFRESIM(SFRDataset):

    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[7], line[3]
        T1 = self.tokenizer(s1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))

class SFRESIM_Weakly(SFRWeaklyZH):
    def __getitem__(self, idx):
        item = self.ori_list[idx]
        s1, s2, label = item[0], item[1], item[2]
        T1 = self.tokenizer(s1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2]), torch.tensor(float(label))