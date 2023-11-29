import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ErnieConfig, ErnieModel
from CC.models.ACErnie.modeling_acernie import ACErnieModel


class ACErnie(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, word_embedding_size=21128, pretrained_embeddings=None):
        super().__init__()
        self.config = ErnieConfig.from_json_file(config_path)
        self.config.output_hidden_states = True
        self.model = ACErnieModel.from_pretrained(
            pre_trained_path, config=self.config)
        self.word_embeddings = nn.Embedding(
            word_embedding_size, self.config.word_embed_dim)
        self.classifier = nn.Linear(self.config.hidden_size * 2, 2)
        self.tokenizer = tokenizer

        self.word_encoder = ErnieModel.from_pretrained(pre_trained_path, config=self.config)

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
        ce_loss = nn.CrossEntropyLoss()

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
        
        max_word_len = self.config.max_word_len
        
        # batch_size * 2 * word_len -> batch_size * word_len
        s1_words = args['sequence_word_ids'][:, 0]
        s2_words = args['sequence_word_ids'][:, 1]
        s1_mask = args['sequence_word_mask'][:, 0]
        s2_mask = args['sequence_word_mask'][:, 1]

        # batch_size * word_len * dim
        s1_words_embeddings = self.word_embeddings(s1_words)
        s2_words_embeddings = self.word_embeddings(s2_words)

        # batch_size * (word_len + 1) * dim
        s1_words_seq_feature = torch.cat([o1.unsqueeze(1), s1_words_embeddings], dim=1)
        s2_words_seq_feature = torch.cat([o2.unsqueeze(1), s2_words_embeddings], dim=1)
        
        # batch_size * (word_len + 1)
        s1_mask_plus = torch.cat([torch.tensor([1]).repeat(s1_mask.shape[0]).unsqueeze(-1).to(s1_mask.device), s1_mask], dim=1)
        s2_mask_plus = torch.cat([torch.tensor([1]).repeat(s2_mask.shape[0]).unsqueeze(-1).to(s2_mask.device), s2_mask], dim=1)

        # batch_size × 2 * (word_len + 1)
        words_seq_feature = torch.cat([s1_words_seq_feature, s2_words_seq_feature], dim=0)
        words_mask = torch.cat([s1_mask_plus, s2_mask_plus], dim=0)

        unsupervised_labels = torch.arange(
            0, words_seq_feature.shape[0], dtype=torch.long).to(words_seq_feature.device)
        
        z1_outputs = self.word_encoder(attention_mask=words_mask, inputs_embeds=words_seq_feature)
        z2_outputs = self.word_encoder(attention_mask=words_mask, inputs_embeds=words_seq_feature)

        z1_cls, z2_cls = torch.sum(z1_outputs[0], dim=1), torch.sum(z2_outputs[0], dim=1)

        sim_matrix = self.cos_sim(z1_cls.unsqueeze(
            1), z2_cls.unsqueeze(0))

        word_loss = ce_loss(sim_matrix, unsupervised_labels)

        u = s1_words_seq_feature * s1_mask_plus.unsqueeze(-1).float()
        u_ = torch.sum(u, dim=1)
        v = s2_words_seq_feature * s2_mask_plus.unsqueeze(-1).float()
        v_ = torch.sum(v, dim=1)
        word_cos_sim = self.cos_sim(u_, v_)

        final_score = (pred + cos_sim) / 2
        
        if 'labels' in args:
            loss = mse_loss(cos_sim, args['labels'].view(-1)) + bce_loss(pred, args['labels'].float().view(-1)) + 0 * mse_loss(word_cos_sim, args['labels'].view(-1) * 0) + 0 * word_loss
        else:
            loss = 0

        return {
            'loss': loss,
            'pred': final_score,
        }
