import os
import torch
from torch.utils.data import DataLoader, Dataset
from CC.ICCStandard import IPredict
from CC.model import AutoModel
from tqdm import tqdm


class Predictor(IPredict):

    def __init__(self, tokenizer, model_name, padding_length=256, batch_size=32, resume_path=False, gpu=0):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.batch_size = batch_size
        self.model_init(tokenizer, model_name)

        device = torch.device("cuda:{}".format(
            gpu) if torch.cuda.is_available() else "cpu")
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

    def __call__(self, sentence_pair_list, input_type='interactive'):
        return self.predict(sentence_pair_list, input_type)

    def predict(self, sentence_pair_list, input_type='interactive'):
        predict_iter = DataLoader(PredictDataset(self.tokenizer, sentence_pair_list, self.padding_length, input_type), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        for it in tqdm(predict_iter):
            for key in it:
                it[key] = self.cuda(it[key])
            outputs = self.model(**it)
            preds = outputs['preds'].tolist()
            yield {
                'preds': preds
            }

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


class PredictDataset():
    def __init__(self, tokenizer, sentence_pair_list, padding_length=256, input_type='interactive'):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.input_type = input_type
        self.sentence_pair_list = sentence_pair_list
        if type(sentence_pair_list[0]) == str:
            self.sentence_pair_list = [sentence_pair_list]

    def get_interactive_input(self, s1, s2):
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length,
                           padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_type_ids': token_type_ids
        }

    def get_simase_input(self, s1, s2):
        left_length = self.padding_length // 2
        if left_length < self.padding_length / 2:
            left_length += 1
        right_length = self.padding_length - left_length
        T1 = self.tokenizer(s1, add_special_tokens=True, max_length=left_length,
                            padding='max_length', truncation=True)
        T2 = self.tokenizer(s2, add_special_tokens=True, max_length=right_length,
                            padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        tid1 = torch.tensor(T1['token_type_ids'])
        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        tid2 = torch.ones(ss2.shape).long()
        return {
            'input_ids': torch.cat([ss1, ss2]),
            'attention_mask': torch.cat([mask1, mask2]),
            'token_type_ids': torch.cat([tid1, tid2])
        }

    def __len__(self):
        return len(self.sentence_pair_list)

    def __getitem__(self, idx):
        item = self.sentence_pair_list[idx]
        s1, s2 = item
        if self.input_type == 'interactive':
            return self.get_interactive_input(s1, s2)
        elif self.input_type == 'siamese':
            return self.get_simase_input(s1, s2)
