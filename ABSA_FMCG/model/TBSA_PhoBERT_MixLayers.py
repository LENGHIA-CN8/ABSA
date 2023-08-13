import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
from transformers import RobertaConfig, RobertaModel
from model.modeling_roberta import RobertaLayer, RobertaPooler
from utils import pool


class PSUM(nn.Module):
    def __init__(self, count, config, num_classes):
        super(PSUM, self).__init__()
        self.count = count
        self.num_classes = num_classes
        self.pre_layers = torch.nn.ModuleList()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
        self.init_weight()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))
        self.drop = nn.Dropout(0.1)

    def init_weight(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.dense.weight.data)
        self.dense.bias.data.uniform_(0, 0)

    def forward(self, layers, attention_mask, target_mask, labels, pool_type, mode, label_weights):
        logitses = []
        losses = []

        for i in range(self.count):
            output = self.pre_layers[i](layers[-i-1], attention_mask)[0]

            if mode == 'pair':
                out = output[:, 0, :]
            elif mode == 'single':
                mask = ~target_mask.unsqueeze(-1).repeat(1,1, output.shape[-1]).bool()
                out = pool(output, mask, pool_type)
            out = self.dense(out)
            out = self.drop(torch.tanh(out))

            logits = self.classifier(out)
            logitses.append(logits)

            if labels is not None:
                ce = nn.CrossEntropyLoss(reduction= 'mean', weight= label_weights)(logits.view(-1, self.num_classes), labels.view(-1))
                losses.append(ce)

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        if labels is not None:
            loss = torch.mean(torch.stack(losses), dim=0)
            return loss, avg_logits

        return avg_logits

class HSUM(nn.Module):
    def __init__(self, count, config, num_classes):
        super(HSUM, self).__init__()
        self.count = count
        self.num_classes = num_classes
        self.pre_layers = torch.nn.ModuleList()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
        self.init_weight()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))
        self.drop = nn.Dropout(0.1)

    def init_weight(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.dense.weight.data)
        self.dense.bias.data.uniform_(0, 0)

    def forward(self, layers, attention_mask, target_mask, labels, pool_type, mode, label_weights):
        logitses = []
        losses = []
        output = torch.zeros_like(layers[0])

        for i in range(self.count):
            output = output + layers[-i-1]
            output = self.pre_layers[i](output, attention_mask)[0]

            if mode == 'pair':
                out = output[:, 0, :]
            elif mode == 'single':
                mask = ~target_mask.unsqueeze(-1).repeat(1,1, output.shape[-1]).bool()
                out = pool(output, mask, pool_type)
            out = self.dense(out)
            out = self.drop(torch.tanh(out))

            logits = self.classifier(out)
            logitses.append(logits)

            if labels is not None:
                ce = nn.CrossEntropyLoss(reduction= 'mean', weight= label_weights)(logits.view(-1, self.num_classes), labels.view(-1))
                losses.append(ce)

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        if labels is not None:
            loss = torch.mean(torch.stack(losses), dim=0)
            return loss, avg_logits
        return avg_logits

class Net(nn.Module):
    def __init__(self, model_dir, config, num_classes, count, mix_type= "HSUM"):
        super(Net, self).__init__()
        self.config = config
        self.phobert = RobertaModel.from_pretrained(model_dir, config=self.config)
        if mix_type.upper() == "HSUM":
            self.mixlayer = HSUM(count, self.config , num_classes)
        elif mix_type.upper() == "PSUM":
            self.mixlayer = PSUM(count, self.config , num_classes)
            
    def forward(self, input_ids, attention_mask, target_mask, labels, mode, pool_type, label_weights):
        layers = self.phobert(input_ids= input_ids, token_type_ids=None, attention_mask=attention_mask)[2]
        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        outputs = self.mixlayer(layers, extend_attention_mask, target_mask, labels, pool_type, mode, label_weights)
        return outputs
