import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification


class R2Bert(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.model = BertForSequenceClassification.from_pretrained(
            pre_trained_path, config=self.config)
        self.tokenizer = tokenizer

    def forward(self, **args):
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        outputs = self.model(
            args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])

        logits = outputs[0]
        p = F.softmax(logits, dim=-1)
        pred = p[:, 1]

        loss = 0.5 * mse_loss(p[:, 1], args['labels'].float().view(-1)) + \
            0.5 * bce_loss(p[:, 1], args['labels'].float().view(-1))

        return {
            'loss': loss,
            'logits': pred,
            'preds': torch.max(p, dim=1)[1]
        }
