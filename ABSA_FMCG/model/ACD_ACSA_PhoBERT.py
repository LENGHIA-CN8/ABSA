import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from transformers import RobertaConfig, RobertaModel
from utils import pool


class cate_cls(nn.Module):
    def __init__(self, config, num_classes):
        super(cate_cls, self).__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.dense.weight.data)
        self.dense.bias.data.uniform_(0, 0)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class Net(nn.Module): 
    def __init__(self, model_path, config, num_classes, num_categories):
        super(Net, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.phobert = RobertaModel.from_pretrained(model_path, config=self.config)
        self.cls_categories = torch.nn.ModuleList()
        for i in range(num_categories):
            self.cls_categories.append(cate_cls(config, num_classes))


    def forward(self, input_ids= None, attention_mask= None, target_mask= None, aspect_type_labels= None, aspect_sentiment_labels= None, 
                mode= 'single', pool_type= 'max', label_weights= [1,1,1,1]):
        out = self.phobert(input_ids= input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
        if mode == 'pair':
            phobert_output = out[:, 0, :]
        elif mode == 'single':
            mask = ~target_mask.unsqueeze(-1).repeat(1,1, out.shape[-1]).bool() # False value will be target
            phobert_output = pool(out, mask, pool_type)
        
        logits = []
        for i, aspect_type_id in enumerate(aspect_type_labels):
            logit = self.cls_categories[aspect_type_id](phobert_output[i])
            logits.append(logit)  
        logits = torch.stack(logits, dim= 0)

        if aspect_sentiment_labels is not None:
            loss = nn.CrossEntropyLoss(reduction= 'mean', weight= label_weights)(logits.view(-1, self.num_classes), aspect_sentiment_labels.view(-1))
            return loss, logits

        return logits


