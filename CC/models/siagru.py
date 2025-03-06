import torch
import torch.nn as nn
import torch.nn.functional as F


class SiaGRU(nn.Module):
    def __init__(self, embeddings=None, hidden_size=300, num_layer=2):
        super(SiaGRU, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        # self.embeds_dim = embeddings.shape[1]
        self.embeds_dim = 300
        # self.word_emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.word_emb = nn.Embedding(21129, 300)
        # self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True
        self.word_emb.to(device)
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0.to(device)
        self.pred_fc = nn.Linear(50, 2)


    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.gru(x)
        return output
    
    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)

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
        
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        sent1, sent2 = args['input_ids'][:, :args['padding_length']], args['input_ids'][:, :args['padding_length']]

        p_encode = self.word_emb(sent1)
        h_endoce = self.word_emb(sent2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        
        encoding1 = self.forward_once(p_encode)
        encoding2 = self.forward_once(h_endoce)
        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        x = self.pred_fc(sim.squeeze(dim=-1))
        probabilities = nn.functional.softmax(x, dim=-1)

        loss = fct_loss(probabilities[:, 1], args['labels'].float())

        pred = probabilities[:,1]
        
        return {
            'loss': loss,
            'logits': pred,
            'preds': torch.max(probabilities, dim=1)[1]
        }