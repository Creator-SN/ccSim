import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ACE(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, num_labels=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(pre_trained_path)
        self.config = AutoConfig.from_pretrained(config_path)
        if num_labels is not None:
            self.config.num_labels = num_labels
        self.cross_attn = nn.MultiheadAttention(self.config.hidden_size, num_heads=4)
        self.pooler_layer = Pooler(self.config)
        self.tokenizer = tokenizer
        
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(1 * self.config.hidden_size, self.config.num_labels)
    
    def contrastive_loss(self, features, labels, temperature=0.1):
        """
        计算带温度的对比学习损失。

        Args:
            features (torch.Tensor): 拼接后的特征矩阵，形状为 [batch_size, 2 * batch_size]。
            labels (torch.Tensor): 标签矩阵，形状为 [batch_size, 2 * batch_size]。
            temperature (float): 温度参数。

        Returns:
            torch.Tensor: 对比学习损失。
        """
        # 计算相似度矩阵
        logits = features / temperature
        log_probs = F.log_softmax(logits, dim=-1)

        # 计算损失
        loss = -torch.sum(labels * log_probs) / labels.shape[0]
        return loss

    def build_contrastive_matrix(self, collect_entity_states_1, collect_entity_states_2):
        """
        构建对比学习矩阵和标签矩阵。

        Args:
            collect_entity_states_1 (torch.Tensor): 形状为 [batch_size, hidden_size]。
            collect_entity_states_2 (torch.Tensor): 形状为 [batch_size, hidden_size]。

        Returns:
            torch.Tensor: 拼接后的特征矩阵，形状为 [batch_size, batch_size]。
            torch.Tensor: 标签矩阵，形状为 [batch_size, batch_size]。
        """
        batch_size = collect_entity_states_1.shape[0]

        # 计算相似性矩阵
        similarity_matrix = F.cosine_similarity(
            collect_entity_states_1.unsqueeze(1),  # [batch_size, 1, hidden_size]
            collect_entity_states_2.unsqueeze(0),  # [1, batch_size, hidden_size]
            dim=-1
        )

        # 构建标签矩阵
        label_matrix = torch.zeros(batch_size, batch_size, device=collect_entity_states_1.device)
        for i in range(batch_size):
            label_matrix[i, i] = i  # 对角线值为行号

        return similarity_matrix, label_matrix
    
    def compute_attn_features(self, text_hidden_states, entity_hidden_states):
        attn_input = torch.stack([text_hidden_states, entity_hidden_states], dim=1)  # [batch, 1+n_ent, hidden]
        attn_output, _ = self.cross_attn(attn_input, attn_input, attn_input)
        text_enhanced = attn_output[:, 0, :]  # 取文本增强表示
        return text_enhanced
    
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

    def forward(self, **args):
        if 'fct_loss' not in args:
            args['fct_loss'] = 'MSELoss'
        t_ids, t_masks, t_types = args['t_ids'], args['t_masks'], args['t_types']
        te_ids, te_masks, te_types = args['te_ids'], args['te_masks'], args['te_types']
        batch_size, seq_len = t_ids.shape[0], t_ids.shape[1]
        input_ids = torch.cat([t_ids, te_ids], dim=0)
        attention_mask = torch.cat([t_masks, te_masks], dim=0)
        token_type_ids = torch.cat([t_types, te_types], dim=0)
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state

        t_h = last_hidden_state[:batch_size, 0, :]
        te_h = last_hidden_state[batch_size : 2 * batch_size, 0, :]

        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # p = F.softmax(logits, dim=-1)
        # pred = p[:, 1]
        
        th_pooled_output = self.pooler_layer(t_h)
        teh_pooled_output = self.pooler_layer(te_h)
        enhanced = self.compute_attn_features(t_h, te_h)
        th_pooled_output = self.dropout(th_pooled_output)
        teh_pooled_output = self.dropout(enhanced)
        logits_1 = self.classifier(th_pooled_output)
        p_1 = F.softmax(logits_1, dim=-1)
        pred = p_1[:, 1]
        logits_2 = self.classifier(teh_pooled_output)
        p_2 = F.softmax(logits_2, dim=-1)

        if 'labels' in args and args['labels'] is not None:
            
            t_cl, t_labels = self.build_contrastive_matrix(t_h, t_h)
            te_cl, te_labels = self.build_contrastive_matrix(te_h, te_h)
            cl_loss_1 = self.contrastive_loss(t_cl, t_labels, 0.05)
            cl_loss_2 = self.contrastive_loss(te_cl, te_labels, 0.05)

            cls_loss_1 = self.compute_loss(self.config.num_labels, args['fct_loss'], logits_1, args['labels'])
            cls_loss_2 = self.compute_loss(self.config.num_labels, args['fct_loss'], logits_2, args['labels'])
            loss = cls_loss_1 + 0.5 * cls_loss_2 + 0.05 * cl_loss_2
        else:
            cls_loss = torch.tensor([0.])
            cl_loss = torch.tensor([0.])
            loss = torch.tensor([0.])

        return {
            'loss': loss,
            'logits': pred
        }