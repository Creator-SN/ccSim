import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ESIM(nn.Module):
    def __init__(self, num_labels=2):
        super(ESIM, self).__init__()
#         self.args = args
        self.num_word = 21129
        self.dropout = 0.5
        self.hidden_size = 300
        self.embeds_dim = 300
        self.linear_size = 200
        self.num_labels = num_labels
        
        self.embeds = nn.Embedding(self.num_word, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.attention = SelfAttention(self.hidden_size*2)
        
        self.fc = nn.Sequential (
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, self.num_labels)
        )

    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, **args):
        if 'fct_loss' not in args:
            args['fct_loss'] = 'MSELoss'
        
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        
        # batch_size * seq_len
        sent1, sent2 = args['input_ids'][:,:args['padding_length']], args['input_ids'][:,args['padding_length']:]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
            
        q1_compose = self.attention(q1_compose)
        q2_compose = self.attention(q2_compose)
        
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        
        similarity = self.fc(x)

        out = similarity
        loss = self.compute_loss(self.num_labels, args['fct_loss'], out, args['labels'])
        
        # 记录准确率
        pred = out[:,1]
        
        return {
            'loss': loss,
            'logits': pred,
            'preds': torch.max(out, dim=1)[1]
        }
    
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
            elif fct_loss == 'CrossEntropyLoss':
                loss = fct_loss(logits, labels.view(-1))

        else:  # num_labels > 2
            if loss_fct != 'CrossEntropyLoss':
                raise ValueError("For num_labels > 2, only CrossEntropyLoss is allowed.")
            loss = fct_loss(logits, labels.view(-1))
        
        return loss
    