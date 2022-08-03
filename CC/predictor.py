import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from CC.ICCStandard import IPredict
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable

class Predictor(IPredict):

    def __init__(self, tokenizer, model_name, padding_length=50, resume_path=False, gpu=0):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.model_init(tokenizer, model_name)

        device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.load_state_dict(model_dict)
        self.model.to(device)
    
    def model_init(self, tokenizer, model_name):
        print('AutoModel Choose Model: {}\n'.format(model_name))
        a = AutoModel(tokenizer, model_name)
        self.model = a()
    
    def data_process(self, X, input_type='bert'):
        f = self.get_bert_input if input_type == 'bert' else self.get_simase_input
        if len(X) < 0:
            raise Exception('The size of input X is zero.')
        elif type(X[0]) == str:
            sentences, attention_mask, token_type_ids = f(X[0], X[1])
            return sentences.unsqueeze(0), attention_mask.unsqueeze(0), token_type_ids.unsqueeze(0)
        else:
            sentences = []
            attention_mask = []
            token_type_ids = []
            for item in X:
                s, a, t = f(item[0], item[1])
                sentences.append(s.unsqueeze(0))
                attention_mask.append(a.unsqueeze(0))
                token_type_ids.append(t.unsqueeze(0))
            return torch.cat(sentences, 0), torch.cat(attention_mask, 0), torch.cat(token_type_ids, 0)
    
    def get_bert_input(self, s1, s2):
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return input_ids, attn_mask, token_type_ids
    
    def get_simase_input(self, s1, s2):
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).int()
        return torch.cat([ss1, ss2]), torch.cat([mask1, mask2]), torch.cat([tid1, tid2])
    
    def __call__(self, X, input_type='bert'):
        return self.predict(X, input_type)
    
    def predict(self, X, input_type='bert'):
        sentences, mask, token_type_ids = self.cuda(self.data_process(X, input_type))
        return self.model(sentences=sentences, attention_mask=mask, token_type_ids=token_type_ids, padding_length=self.padding_length)
    
    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX