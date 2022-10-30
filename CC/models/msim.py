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

class SequenceDiff(nn.Module):

    def __init__(self, config, slider_size=6):
        super().__init__()
        self.config = config
        self.slider_size = slider_size
        self.lstm_hidden_size = int(self.config.hidden_size / 2)
        self.lstm = nn.LSTM(
            config.hidden_size * 2, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2 *
                      (config.num_attention_heads + 1), 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, sentences, attentions, logits, mask):
        # sentences: batch_size, seq_len
        # logits: batch_size, seq_len, hidden_size
        # attentions: batch_size, num_heads, sequence_length, sequence_length
        # mask: batch_size, seq_len
        # print(attentions.shape)
        # print(0 / 0)

        self.lstm.flatten_parameters()

        # This operation aims to clear the influence caused by the [SEP]. Accoring to Clark K, Khandelwal U, Levy O, et al. What does bert look at? an analysis of bert's attention[J]. arXiv preprint arXiv:1906.04341, 2019.
        attentions = attentions.clone().transpose(1, 3)
        attentions[sentences == 102] = 1e-9
        attentions = attentions.clone().transpose(1, 3)

        seq_len = int(attentions.shape[2] / 2)
        A, B = attentions[:, :, :seq_len,
                          :seq_len], attentions[:, :, seq_len:, seq_len:]
        maskA, maskB = mask[:, :seq_len], mask[:, seq_len:]

        slide_seq = torch.tensor([])
        if torch.cuda.is_available():
            slide_seq = slide_seq.cuda()

        # calculate the sum by each column
        sum_A = torch.sum(A.transpose(2, 3), 2)
        sum_B = torch.sum(B.transpose(2, 3), 2)

        # masterA: batch_size * num_heads
        masterA = torch.max(sum_A, -1)[1]
        masterB = torch.max(sum_B, -1)[1]
        for i in range(masterA.shape[0]):

            for j in range(masterA.shape[1]):
                pos = masterA[i][j].data.item()
                pos, l, r = self.get_range_index(pos, seq_len)
                s_a = logits[i, pos - l: pos + r, :]
                s_mask_a = maskA[i, pos - l: pos + r]
                s_mask_a = s_mask_a.unsqueeze(-1).expand(s_a.size())
                s_a[s_mask_a == 0] = 1e-9

                pos = masterB[i][j].data.item()
                pos, l, r = self.get_range_index(pos, seq_len)
                s_b = logits[i, seq_len + pos - l: seq_len + pos + r, :]
                s_mask_b = maskB[i, pos - l: pos + r]
                s_mask_b = s_mask_b.unsqueeze(-1).expand(s_b.size())
                s_b[s_mask_b == 0] = 1e-9

                s_ab = torch.cat([s_a, s_b], -1)
                # slide_seq: batch_size * num_heads * slider_size, hidden_size * 2
                slide_seq = torch.cat([slide_seq, s_ab], -2)

        # slide_seq: batch_size * num_heads, slider_size, hidden_size * 2
        slide_seq = slide_seq.view(
            masterA.shape[0] * masterA.shape[1], self.slider_size, -1)

        # compose: batch_size * num_heads, slider_size, 2 * lstm_hidden_size
        compose, _ = self.lstm(slide_seq)

        # rep: batch_size * num_heads, 2 * lstm_hidden_size
        rep = self.pooling(compose)
        # logits_rep: batch_size, 2 * lstm_hidden_size
        logits_rep = self.pooling(logits)

        # batch_rep: batch_size, num_heads, 2 * lstm_hidden_size
        batch_rep = rep.reshape(masterA.shape[0], masterA.shape[1], -1)
        # x: batch_size, num_heads * 2 * lstm_hidden_size
        x = batch_rep.view(masterA.shape[0], -1)
        x = torch.cat([x, logits_rep], -1)

        similarity = self.fc(x)

        return similarity

    def pooling(self, x):
        # transpose(1,2): take average of per index of hidden states in different words
        o1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        # o1: batch_size, 2 * lstm_hidden_size
        return o1

    def get_range_index(self, pos, seq_len):

        pos = 1 if pos == 0 else pos
        pos = seq_len - 2 if pos == seq_len - 1 else pos

        l = int(self.slider_size / 2)
        if pos - l <= 0:
            l = pos - 1
        r = self.slider_size - l
        if pos + r >= seq_len - 1:
            r = seq_len - pos - 2
            l = self.slider_size - r

        return pos, l, r



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
        self.sd = SequenceDiff(self.config, slider_size=8)
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

        # Different Layers Attentions
        attention_list = outputs[3]

        # keywords-level
        sim = self.sd(args['input_ids'], attention_list[-1],
                          logits=logits, mask=args['attention_mask'].long())
        predA = sim[:, 1]

        hs_1 = hidden_states[0][:, :args['padding_length'], :]
        hs_2 = hidden_states[0][:, args['padding_length']:, :]

        w1_input_feature = self.word_embedding(args['words_input_ids_1'])
        w2_input_feature = self.word_embedding(args['words_input_ids_2'])

        # keywords-level
        word_feature_1_out = self.wa(
            hs_1[:, 0, :], w2_input_feature, args['words_attention_mask_2'])
        word_feature_2_out = self.wa(
            hs_2[:, 0, :], w1_input_feature, args['words_attention_mask_1'])

        # sentence-level
        predB, cs_feature = self.cs(logits_1, args['attention_mask'][:, :args['padding_length']],
                                    logits_2, args['attention_mask'][:, args['padding_length']:])

        # mix_output = torch.cat(
        #     [word_feature_1_out, word_feature_2_out, cs_feature], dim=1)
        
        # out = self.fc(mix_output)
        pred = 0.1 * predA + 1 * predB + 0.2 * (word_feature_1_out + word_feature_2_out)

        if not args['labels'] is None:
            loss = fct_loss(pred, args['labels'].view(-1))
            return loss, pred, (predA, predB, word_feature_1_out, word_feature_2_out)

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
