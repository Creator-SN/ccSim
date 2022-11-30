import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, embedding_dim=300):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(21129, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, embedding_dim)) for k in [2, 3, 4]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * len([2, 3, 4]), 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

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
        
        out = self.embedding(args['input_ids'])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                        for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        loss = fct_loss(out[:, 1], args['labels'].float())
        pred = out[:,1]

        return {
            'loss': loss,
            'pred': pred
        }
