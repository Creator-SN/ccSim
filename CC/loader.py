import os
import json
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.ICCStandard import IDataLoader
from CC.loaders import *


class AutoDataloader(IDataLoader):

    '''
    loader_name: str; the dataloader name
    data_path: str or obj; the path of the data; if str, it will use the present dataset in data_present_path, or you should define the path like e.g. { 'train': './train.json', 'dev': './dev.json' }
    model_type: interactive or siamese
    data_present_path: str; the path of the data_present; the data_present is a json file which contains the path of the dataset, and the format is like e.g. { 'dataset_name': {'train': './train.json', 'dev': './dev.json'} }
    padding_length: int; the length of the padding
    '''

    def __init__(self, tokenizer, loader_name='CNSTS', data_path="CNSTS", model_type="interactive", data_present_path="./dataset/present.json", padding_length=50):
        self.loader_name = loader_name
        self.model_type = model_type
        self.padding_length = padding_length
        self.data_present = self.get_data_present(data_present_path)
        self.data_path = self.data_present[data_path] if data_path in self.data_present else data_path
        if loader_name == 'CNSTS':
            self.train_set = CNSTSDataset(
                tokenizer, self.data_path['train'], padding_length=self.padding_length, model_type=self.model_type, shuffle=True)
            self.eval_set = CNSTSDataset(
                tokenizer, self.data_path['dev'], padding_length=self.padding_length, model_type=self.model_type, shuffle=False)
            if 'test' in self.data_path:
                self.test_set = CNSTSDataset(
                    tokenizer, self.data_path['test'], padding_length=self.padding_length, model_type=self.model_type, shuffle=False)
        elif loader_name == 'CNSTSX':
            self.train_set = CNSTSXDataset(
                tokenizer, self.data_path['train'], self.data_path['vocab_file'], padding_length=self.padding_length, model_type=self.model_type, shuffle=True)
            self.eval_set = CNSTSXDataset(
                tokenizer, self.data_path['dev'], self.data_path['vocab_file'], padding_length=self.padding_length, model_type=self.model_type, shuffle=False)
            if 'test' in self.data_path:
                self.test_set = CNSTSXDataset(
                    tokenizer, self.data_path['test'], self.data_path['vocab_file'], padding_length=self.padding_length, model_type=self.model_type, shuffle=False)
        elif loader_name == "SIMCSE_STS":
            # just copy first sentence twice.
            self.train_set = CNSTSXDataset(
                tokenizer, self.data_path["train"], self.data_path['vocab_file'], padding_length=self.padding_length, shuffle=True, model_type="copy"
            )
            self.eval_set = CNSTSXDataset(
                tokenizer, self.data_path["train"], self.data_path['vocab_file'], padding_length=self.padding_length, shuffle=True, model_type=self.model_type
            )
            if 'test' in self.data_path:
                self.test_set = CNSTSXDataset(
                    tokenizer, self.data_path['test'], self.data_path['vocab_file'], padding_length=self.padding_length, model_type=self.model_type, shuffle=False)

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

    def __call__(self, batch_size=16, batch_size_eval=64, eval_mode='dev'):
        dataiter = DataLoader(self.train_set, batch_size=batch_size)
        if eval_mode == 'dev':
            dataiter_eval = DataLoader(
                self.eval_set, batch_size=batch_size_eval)
        else:
            dataiter_eval = DataLoader(
                self.test_set, batch_size=batch_size_eval)
        return dataiter, dataiter_eval
