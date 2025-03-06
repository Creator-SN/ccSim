import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)

#         outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        outputs = (encoder_outputs * weights)

#         return outputs, weights
        return outputs


class CosineLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config.hidden_size)
        Cuda(self.attention)
        self.cosine_score_transformation = nn.Identity(torch.cosine_similarity)

    def pooling(self, x, mask, mode='mean'):
        # input: batch_size * seq_len * hidden_size
        cls_token = x[:, 0, :]
        output_vectors = []
        if mode == 'cls':
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

    def forward(self, em1, mask1, em2, mask2, labels):
        mse = nn.MSELoss()

        em1 = self.attention(em1)
        em2 = self.attention(em2)

        u = self.pooling(em1, mask1)
        v = self.pooling(em2, mask2)

        # output: batch_size * (9 * hidden_size)
        # mixed = torch.cat([u, v, torch.abs(u - v)], 1)
        # similarity = self.softmax(mixed)
        similarity = self.cosine_score_transformation(
            torch.cosine_similarity(u, v))
        return similarity, mse(similarity, labels)


class WSSBert(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.model = BertModel.from_pretrained(
            pre_trained_path, config=self.config)
        self.fct_loss = CosineLoss(self.config)
        self.tokenizer = tokenizer

    def forward(self, **args):

        Cuda(self.fct_loss)
        outputs = self.model(
            args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])[0]

        pred, loss = self.fct_loss(outputs[:, :args['padding_length'], :], args['attention_mask'][:, :args['padding_length']],
                                   outputs[:, args['padding_length']:, :], args['attention_mask'][:, args['padding_length']:], args['labels'].view(-1))

        return {
            'loss': loss,
            'logits': pred,
            'preds': (pred > 0.5).long()
        }

    def get_model(self):
        return self.model


def Cuda(object: nn.Module):
    if torch.cuda.is_available():
        object.cuda()
