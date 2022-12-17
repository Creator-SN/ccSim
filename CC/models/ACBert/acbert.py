import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from CC.models.ACBert.modeling_acbert import ACBertModel


class ACBert(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, word_embedding_size=21128, pretrained_embeddings=None):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.model = ACBertModel.from_pretrained(
            pre_trained_path, config=self.config)
        self.word_embeddings = nn.Embedding(
            word_embedding_size, self.config.word_embed_dim)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.tokenizer = tokenizer

        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))

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

        args['matched_word_embeddings'] = self.word_embeddings(
            args['matched_word_ids'])

        outputs = self.model(args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'],
                             matched_word_embeddings=args['matched_word_embeddings'], matched_word_mask=args['matched_word_mask'])

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        
        p = F.softmax(logits, dim=-1)
        pred = p[:, 1]

        if 'labels' in args:
            loss = fct_loss(p[:, 1], args['labels'].view(-1))
        else:
            loss = 0

        return {
            'loss': loss,
            'pred': pred,
        }
