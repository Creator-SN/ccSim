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


class CosineSim(nn.Module):
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

    def forward(self, em1, mask1, em2, mask2):

        em1 = self.attention(em1)
        em2 = self.attention(em2)

        u = self.pooling(em1, mask1)
        v = self.pooling(em2, mask2)

        # output: batch_size * (9 * hidden_size)
        mixed = torch.cat([u, v, torch.abs(u - v)], 1)
        # similarity = self.softmax(mixed)
        similarity = self.cosine_score_transformation(
            torch.cosine_similarity(u, v))
        return similarity, mixed


class WordAttentionFeature(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.Tanh()

        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, bidirectional=True, batch_first=True)
        self.word_transform = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.word_word_weight = nn.Linear(
            config.hidden_size, config.hidden_size)
        attn_W = torch.zeros(config.hidden_size, config.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)
        self.fuse_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, sentence, words, words_attention_mask):
        # words: batch_size * vocab_size * hidden_size
        words_feautre, _ = self.lstm(words)
        word_outputs = self.word_transform(
            words_feautre)  # [N, W, D]
        word_outputs = self.act(word_outputs)
        word_outputs = self.word_word_weight(word_outputs)
        word_outputs = self.dropout(word_outputs)

        # sentence: [N, D] -> [N, 1, D]
        alpha = torch.matmul(sentence.unsqueeze(1),
                             self.attn_W)  # [N, 1, D]
        alpha = torch.matmul(alpha, torch.transpose(
            word_outputs, 1, 2))  # [N, 1, W]
        alpha = alpha.squeeze(1)  # [N, W]
        alpha = alpha + (1 - words_attention_mask.float()) * (-10000.0)
        alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, W]
        alpha = alpha.unsqueeze(-1)  # [N, W, 1]
        weighted_word_embedding = torch.sum(
            word_outputs * alpha, dim=1)  # [N, D]
        word_attn_feature = weighted_word_embedding

        word_attn_feature = self.dropout(word_attn_feature)
        # word_attn_feature = self.fuse_layernorm(word_attn_feature)

        sim = self.fc(torch.cat([word_attn_feature, sentence], dim=-1))

        return sim[:, 1]


class MSIM(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, mode='default'):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.output_attentions = True
        self.config.output_hidden_states = True
        self.model = BertModel.from_pretrained(
            pre_trained_path, config=self.config)
        self.word_embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size)
        self.mode = mode
        self.wa = WordAttentionFeature(self.config)
        self.cs = CosineSim(self.config)
        self.fc = nn.Sequential(
            nn.Linear(5 * self.config.hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        self.mode_update()
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

        Cuda(self.cs)
        Cuda(self.wa)
        outputs = self.model(
            args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])

        logits = outputs[0]

        logits_1 = logits[:, :args['padding_length'], :]
        logits_2 = logits[:, args['padding_length']:, :]

        # Different Layers Attentions
        hidden_states = outputs[2]

        hs_1 = hidden_states[3][:, :args['padding_length'], :]
        hs_2 = hidden_states[3][:, args['padding_length']:, :]

        w1_input_feature = self.word_embedding(args['words_input_ids_1'])
        w2_input_feature = self.word_embedding(args['words_input_ids_2'])

        # keywords-level
        word_feature_1_out = self.wa(
            torch.sum(hs_1, dim=1), w2_input_feature, args['words_attention_mask_2'])
        word_feature_2_out = self.wa(
            torch.sum(hs_2, dim=1), w1_input_feature, args['words_attention_mask_1'])

        # sentence-level
        predB, cs_feature = self.cs(logits_1, args['attention_mask'][:, :args['padding_length']],
                                    logits_2, args['attention_mask'][:, args['padding_length']:])

        # mix_output = torch.cat(
        #     [word_feature_1_out, word_feature_2_out, cs_feature], dim=1)
        
        # out = self.fc(mix_output)
        pred = predB + word_feature_1_out + word_feature_2_out

        if not args['labels'] is None:
            loss = fct_loss(pred, args['labels'].view(-1))
            return loss, pred

        return pred

    def get_model(self):
        return self.model

    def mode_update(self):
        if self.mode == 'keywords_only':
            self.cs.eval()
            self.wa.train()
        elif self.mode == 'seq_only':
            self.cs.train()
            self.wa.eval()
        else:
            self.cs.train()
            self.wa.train()


def Cuda(object: nn.Module):
    if torch.cuda.is_available():
        object.cuda()
