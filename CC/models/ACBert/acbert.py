import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from CC.models.ACBert.modeling_acbert import ACBertModel


class ACBert(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, word_embedding_size=21128, pretrained_embeddings=None):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.output_hidden_states = True
        self.model = ACBertModel.from_pretrained(
            pre_trained_path, config=self.config)
        self.word_embeddings = nn.Embedding(
            word_embedding_size, self.config.word_embed_dim)
        self.classifier = nn.Linear(self.config.hidden_size * 2, 2)
        self.tokenizer = tokenizer

        self.cos_sim = nn.CosineSimilarity(dim=-1)

        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))
    
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
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

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

        last_hidden_state = outputs.last_hidden_state

        length = args['padding_length']
        o1 = last_hidden_state[:, 0, :]
        o2 = last_hidden_state[:, length, :]
        o = torch.cat([o1, o2], dim=-1)

        logits = self.classifier(o)
        
        p = F.softmax(logits, dim=-1)
        pred = p[:, 1]

        h1 = outputs.hidden_states[1]
        h1_1 = h1[:, :length, :]
        mask1 = args['attention_mask'][:, :length]
        h1_2 = h1[:, length:, :]
        mask2 = args['attention_mask'][:, length:]

        u = self.pooling(h1_1, mask1)
        v = self.pooling(h1_2, mask2)
        cos_sim = self.cos_sim(u, v)

        final_score = (pred + cos_sim) / 2
        
        if 'labels' in args:
            loss = mse_loss(cos_sim, args['labels'].view(-1)) + bce_loss(pred, args['labels'].float().view(-1))
        else:
            loss = 0

        return {
            'loss': loss,
            'pred': final_score,
        }
