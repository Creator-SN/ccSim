from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class SIMCSE(nn.Module):

    def __init__(self, tokenizer, config_path, pre_trained_path, temp=0.05):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.model = BertModel.from_pretrained(
            pre_trained_path, config=self.config)
        self.tokenizer = tokenizer
        self.lossfct = nn.CrossEntropyLoss()
        self.sim = nn.CosineSimilarity(dim=-1)
        self.temp = temp

    def forward(self, **args):
        if args["input_ids"].shape[-1] % 2 != 0:
            raise Exception("input_ids shape couldn't divide 2")
        # input_ids: [batch,2*seq_length]
        seq_length = args["input_ids"].shape[-1]//2
        # suppose z1 is similar to z2
        z1, z2 = args["input_ids"][:,
                                   :seq_length], args["input_ids"][:, seq_length:]

        unsupervised_labels = torch.arange(
            0, z1.shape[0], dtype=torch.long).to(z1.device)

        z1_outputs = self.model(
            z1, attention_mask=args["attention_mask"][:, :seq_length], token_type_ids=args["token_type_ids"][:, :seq_length])
        z2_outputs = self.model(
            z2, attention_mask=args["attention_mask"][:, seq_length:], token_type_ids=args["token_type_ids"][:, seq_length:])

        z1_cls, z2_cls = z1_outputs[0][:, 0], z2_outputs[0][:, 0]

        sim_matrix = self.sim(z1_cls.unsqueeze(
            1), z2_cls.unsqueeze(0))/self.temp

        loss = self.lossfct(sim_matrix, unsupervised_labels)

        pred = self.sim(z1_cls, z2_cls)


        return {
            'loss': loss,
            'pred': pred
        }
