import torch
from time import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from evaluate.evaluate_acd_acsa import mi_ma_cro_f1, get_confusion_matrix, get_strict_acc_acd, get_strict_acc_acd_acsa
from utils import get_model, set_model_
from transformers import AdamW, get_linear_schedule_with_warmup
from optimization import BertAdam


def warmup_linear(x, warmup= 0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

class Trainer():
    def __init__(self, args, model, train_loader, valid_loader, test_loader, log_file, device= 'cuda'):
        super(Trainer, self).__init__()
        self.model = model.to(device)
        self.device = device
        self.log_file = log_file
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.label_weights = torch.Tensor(args.label_weights).to(self.device)
        self.optimizer = self.get_optimizer()
        
        self.t_total = int(len(self.train_loader) * self.args.epochs / self.args.gradient_accumulation_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps= args.warmup_steps, num_training_steps= self.t_total)

    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if self.args.optim == "AdamW":
            return AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        elif self.args.optim == "BertAdam":
            return BertAdam(optimizer_grouped_parameters, lr=self.args.lr, warmup= self.args.warmup_steps, 
                            t_total=self.t_total)
        else:
            raise Exception('Have not implement {} optimizer for this model yet'.format(self.args.optim))

    def train(self):
        global_step = 0
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []
        self.best_valid_macro_f1 = -1e3
        self.best_model = get_model(self.model)

        for epoch in range(self.args.epochs):
            # train_loss = 0
            # train_acc = 0
            # train_steps = 0
            # train_groundtruth = 0

            time1 = time()
            self.model.zero_grad()
            for step, batch in enumerate(self.train_loader):
                self.model.train()
                batch = [t.to(self.device) for t in batch]
                input_ids, attention_mask, token_type_ids, target_masks, aspect_type_labels, aspect_sentiment_labels = batch

                loss, logits = self.model(input_ids, attention_mask, target_masks, aspect_type_labels, aspect_sentiment_labels, 
                                          self.args.data_mode, self.args.pool_type, self.label_weights)
                
                if self.args.gradient_accumulation_steps > 1: #accumulate gradient
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()

                # out_prob = F.softmax(logits.view(-1, self.args.num_classes), 1)
                # out_prob = out_prob.detach().cpu().numpy()
                # label_ids = label.view(-1).to('cpu').numpy()
                # preds = np.argmax(out_prob, axis=1)
                # tmp_train_accuracy = (preds == label_ids)

                # train_loss += loss.item()
                # train_acc += np.sum(tmp_train_accuracy)

                # train_groundtruth += len(input_ids)
                # train_steps += 1
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                 
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

            # self.scheduler.step()
            time2 = time()
            
            valid_loss, valid_acc, valid_micro_f1, valid_macro_f1, valid_confusion, valid_strict_acc_acd, valid_strict_acc_acd_acsa = self.eval(self.model, self.valid_loader)
            train_loss, train_acc, _, _, _, _, _ =  self.eval(self.model, self.train_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
          

            if valid_macro_f1 > self.best_valid_macro_f1:
                self.best_valid_macro_f1 = valid_macro_f1
                self.best_model = get_model(self.model)

            print("Epoch {:3d} | Train: loss= {:.3f}, acc= {:.3f}% || Valid: loss= {:.3f}, acc= {:.3f}%, micro_f1= {:.3f}%, macro_f1= {:.3f}%, strict_acc_acd= {:.3f}%, strict_acc_acd_acsa= {:.3f}% || Best macro_f1: {:.3f} || Time= {:.3f}s".format(epoch + 1, train_loss, train_acc*100, valid_loss, valid_acc*100, valid_micro_f1*100, valid_macro_f1*100, valid_strict_acc_acd*100, valid_strict_acc_acd_acsa*100, self.best_valid_macro_f1*100, time2- time1), file= self.log_file)
    
            print("Epoch {:3d} | Train: loss= {:.3f}, acc= {:.3f}% || ".format(epoch + 1, train_loss, train_acc), end= "")
            print("Valid: loss= {:.3f}, acc= {:.3f}%, micro_f1= {:.3f}%, macro_f1= {:.3f}%, strict_acc_acd= {:.3f}%, strict_acc_acd_acsa= {:.3f}% || Best macro_f1: {:.3f} || Time= {:.3f}s ||".format(valid_loss, valid_acc*100, valid_micro_f1*100, valid_macro_f1*100, valid_strict_acc_acd*100, valid_strict_acc_acd_acsa*100, self.best_valid_macro_f1*100, time2- time1), end= "\n")
        
        set_model_(self.model, self.best_model) # set back to the best model found

    def eval(self, model, data):
        total_loss = 0
        acc = 0
        steps = 0
        groundtruth_count = 0

        prediction = []
        groundtruths = []
        bin_masks = []

        for step, batch in enumerate(data):
            model.eval()
            batch = [t.to(self.device) for t in batch]
            input_ids, attention_mask, token_type_ids, target_masks, aspect_type_labels, aspect_sentiment_labels = batch

            loss, logits = self.model(input_ids, attention_mask, target_masks, aspect_type_labels, aspect_sentiment_labels, 
                                      self.args.data_mode, self.args.pool_type, self.label_weights)

            out_prob = F.softmax(logits.view(-1, self.args.num_classes), 1)
            out_prob = out_prob.detach().cpu().numpy()
            label_ids = aspect_sentiment_labels.view(-1).to('cpu').numpy()
            preds = np.argmax(out_prob, axis=1)
            tmp_accuracy = (preds == label_ids)

            total_loss += loss.item()
            acc += np.sum(tmp_accuracy)

            steps += 1
            groundtruth_count += len(input_ids)

            prediction += list(preds)
            groundtruths += list(label_ids)

        loss, acc = total_loss/ steps, acc/groundtruth_count
        micro, macro = mi_ma_cro_f1(groundtruths, prediction)
        confusion_matrix = get_confusion_matrix(groundtruths, prediction, self.args.aspect_sentiment_dict)
        strict_acc_acd = get_strict_acc_acd(groundtruths, prediction, self.args.aspect_type_dict, self.args.aspect_sentiment_dict)
        strict_acc_acd_acsa = get_strict_acc_acd_acsa(groundtruths, prediction, self.args.aspect_type_dict)
        return loss, acc, micro, macro, confusion_matrix, strict_acc_acd, strict_acc_acd_acsa
