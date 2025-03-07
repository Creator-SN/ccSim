import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification

class Bert(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, num_labels=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(config_path)
        if num_labels is not None:
            self.config.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(pre_trained_path, config=self.config)
        self.tokenizer = tokenizer
    
    def compute_loss(self, num_labels: int, loss_fct: str, logits, labels=None):
        if labels is None:
            return torch.tensor(0).to(logits.device)
        fct_loss = nn.MSELoss()
        if loss_fct == 'BCELoss':
            fct_loss = nn.BCELoss()
        elif loss_fct == 'CrossEntropyLoss':
            fct_loss = nn.CrossEntropyLoss()
        elif loss_fct == 'MSELoss':
            fct_loss = nn.MSELoss()

        if num_labels == 1:
            if loss_fct not in ['BCELoss', 'MSELoss']:
                raise ValueError("For num_labels == 1, only BCELoss and MSELoss are allowed.")
            pred = torch.sigmoid(logits)
            loss = fct_loss(pred.view(-1), labels.view(-1).float())     

        elif num_labels == 2:
            if loss_fct in ['BCELoss', 'MSELoss']:
                p = F.softmax(logits, dim=-1)
                pred = p[:, 1]
                loss = fct_loss(pred, labels.view(-1).float())
            elif loss_fct == 'CrossEntropyLoss':
                loss = fct_loss(logits, labels.view(-1))

        else:  # num_labels > 2
            if loss_fct != 'CrossEntropyLoss':
                raise ValueError("For num_labels > 2, only CrossEntropyLoss is allowed.")
            loss = fct_loss(logits, labels.view(-1))
        
        return loss
    
    def forward(self, **args):
        if 'fct_loss' not in args:
            args['fct_loss'] = 'MSELoss'
        
        outputs = self.model(args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])

        logits = outputs[0]
        p = F.softmax(logits, dim=-1)
        pred = p[:, 1]

        if 'labels' in args:
            loss = self.compute_loss(self.config.num_labels, args['fct_loss'], logits, args['labels'])
        else:
            loss = 0

        return {
            'loss': loss,
            'logits': pred,
            'preds': torch.max(p, dim=1)[1]
        }