import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from transformers import RobertaConfig, RobertaModel
from utils import pool

class Net(nn.Module): 
    def __init__(self, model_path, config, num_classes):
        super(Net,self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.phobert = RobertaModel.from_pretrained(model_path, config=self.config)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weight()

    def init_weight(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.dense.weight.data)
        self.dense.bias.data.uniform_(0, 0)
        
    def forward(self, input_ids, attention_mask, target_mask, labels= None, mode= 'single', pool_type= 'max', label_weights= [1,1,1,1]):
        out = self.phobert(input_ids= input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
        if mode == 'pair':
            phobert_output = out[:, 0, :]
        elif mode == 'single':
            mask = ~target_mask.unsqueeze(-1).repeat(1,1, out.shape[-1]).bool() # False value will be target
            phobert_output = pool(out, mask, pool_type)
        
        x = self.dropout(phobert_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss(reduction= 'mean', weight= label_weights)(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits

        return logits


