import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM

class BertLM(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.model = BertForMaskedLM.from_pretrained(pre_trained_path, config=self.config)
        self.tokenizer = tokenizer
    
    def forward(self, **args):
        outputs = self.model(args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'], labels=args['labels'])

        loss = outputs[0]

        return {
            'loss': loss
        }