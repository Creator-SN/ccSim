import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetConfig, XLNetForSequenceClassification

class XLNet(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path):
        super().__init__()
        self.config = XLNetConfig.from_json_file(config_path)
        self.model = XLNetForSequenceClassification.from_pretrained(pre_trained_path, config=self.config)
        self.tokenizer = tokenizer
    
    def forward(self, **args):
        if 'fct_loss' in args:
            if args['fct_loss'] == 'BCELoss':
                fct_loss = nn.BCELoss()
            elif args['fct_loss'] == 'CrossEntropyLoss':
                fct_loss = nn.CrossEntropyLoss()
            elif args['fct_loss'] == 'MSELoss':
                fct_loss = nn.MSELoss()
        else:
            fct_loss = nn.MSELoss()
        
        outputs = self.model(args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])

        logits = outputs[0]
        p = F.softmax(logits, dim=-1)
        pred = p[:, 1]

        loss = fct_loss(p[:, 1], args['labels'].view(-1))

        return {
            'loss': loss,
            'logits': pred,
            'preds': torch.max(p, dim=1)[1]
        }