import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

class SBert(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.model = BertModel.from_pretrained(pre_trained_path, config=self.config)
        self.tokenizer = tokenizer

        self.hidden_size = self.config.hidden_size
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 2),
            nn.Softmax(dim=-1)
        )
        self.cosine_score_transformation = nn.Identity(torch.cosine_similarity)
    
    def pooling(self, x, mask, mode = 'mean'):
        # input: batch_size * seq_len * hidden_size
        cls_token = x[:, 0, :]
        output_vectors = []
        if mode =='cls':
            output_vectors.append(cls_token)
        elif mode == 'max':
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            x[input_mask_expanded == 0] = -1e9
            max_over_timer = torch.max(x, 1)[0]
            output_vectors.append(max_over_timer)
        elif mode == 'mean':
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        # print('cls_token', cls_token.shape, 'input_mask_expanded', input_mask_expanded.shape, 'sum_mask', sum_mask.shape)
        # print(0 / 0)
        return output_vector
    
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
        x1, mask1, tid1 = args['input_ids'][:, :args['padding_length']], args['attention_mask'][:, :args['padding_length']], args['token_type_ids'][:, :args['padding_length']]
        x2, mask2, tid2 = args['input_ids'][:, args['padding_length']:], args['attention_mask'][:, args['padding_length']:], args['token_type_ids'][:, args['padding_length']:]
        em1 = self.model(x1, attention_mask=mask1, token_type_ids=tid1)[0]
        em2 = self.model(x2, attention_mask=mask2, token_type_ids=tid2)[0]

        u = self.pooling(em1, mask1)
        v = self.pooling(em2, mask2)
        similarity = self.cosine_score_transformation(torch.cosine_similarity(u, v))

        loss = fct_loss(similarity, args['labels'].float())
        
        # 记录准确率
        pred = similarity

        return {
            'loss': loss,
            'pred': pred,
        }
    
    def get_model(self):
        return self.model