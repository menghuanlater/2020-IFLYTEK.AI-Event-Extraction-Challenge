#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020-07-23
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage: 
-----------------------------
    Description:
    各论元识别存在与否的分类损失权重
    1. object分类损失权重: [1.0, 1.0]
    2. subject分类损失权重: [10, 0.6]
    3. time分类损失权重: [0.76, 1.45]
    4. location分类损失权重: [0.6, 5.5]
-----------------------------
"""
from transformers import BertTokenizer, BertModel
import torch
import pickle
import sys
import datetime
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, tokenizer: BertTokenizer, max_len, special_query_token_map: dict):
        self.data = data
        self.map = special_query_token_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        context, query, answer, _type = item["context"], item["query"], item["answer"], item["type"]
        # 首先编码input_ids ==> 分为Q和P两部分
        query_tokens = []
        for i in query:
            if i in self.map.keys():
                query_tokens.append(self.map[i])
            else:
                query_tokens.append(i)
        context_tokens = [i for i in context]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len-1]
        c += ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        x = len(query_tokens) + 2
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_seg": torch.tensor(input_seg).long(),
            "input_mask": torch.tensor(input_mask).float(), "context": context,
            "context_range": "%d-%d" % (2 + len(query_tokens), context_end),  # 防止被转化成tensor
            "cls": answer["is_exist"], "label": answer["argument"],
            "start_index": x + answer["start"], "end_index": x + answer["end"],
            "object_mask": 1.0 if _type == "object" else 0.0, "subject_mask": 1.0 if _type == "subject" else 0.0,
            "time_mask": 1.0 if _type == "time" else 0.0, "location_mask": 1.0 if _type == "location" else 0.0,
            "type": _type
        }


class MyModel(torch.nn.Module):
    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_rate)
        )
        self.cls_layer = torch.nn.Linear(in_features=1024, out_features=2)
        self.start_layer = torch.nn.Linear(in_features=1024, out_features=1)
        self.end_layer = torch.nn.Linear(in_features=1024, out_features=1)
        self.object_cls_lfc = torch.nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([1.0, 1.0]).float().to(device))
        self.subject_cls_lfc = torch.nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([10.0, 0.6]).float().to(device))
        self.time_cls_lfc = torch.nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([0.76, 1.45]).float().to(device))
        self.location_cls_lfc = torch.nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([0.6, 5.5]).float().to(device))
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg, cls_label=None, start_index=None, end_index=None,
                object_mask=None, subject_mask=None, time_mask=None, location_mask=None):
        encoder_rep, cls_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[:2]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        cls_logits = self.cls_layer(cls_rep)
        start_logits = self.start_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        end_logits = self.end_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        # adopt softmax function across length dimension with masking mechanism
        mask = input_mask == 0.0
        start_logits.masked_fill_(mask, -1e30)
        end_logits.masked_fill_(mask, -1e30)
        start_prob_seq = torch.nn.functional.softmax(start_logits, dim=1)
        end_prob_seq = torch.nn.functional.softmax(end_logits, dim=1)
        if start_index is None or end_index is None or cls_label is None:
            return cls_logits, start_prob_seq, end_prob_seq
        else:
            object_loss = self.object_cls_lfc(input=cls_logits, target=cls_label)
            subject_loss = self.subject_cls_lfc(input=cls_logits, target=cls_label)
            time_loss = self.time_cls_lfc(input=cls_logits, target=cls_label)
            location_loss = self.location_cls_lfc(input=cls_logits,target=cls_label)
            cls_loss = object_loss * object_mask + subject_loss * subject_mask + time_loss * time_mask + location_loss * location_mask
            # indices select
            start_prob = (start_prob_seq.gather(index=start_index.unsqueeze(dim=-1), dim=1) + self.epsilon).squeeze(dim=-1)
            end_prob = (end_prob_seq.gather(index=end_index.unsqueeze(dim=-1), dim=1) + self.epsilon).squeeze(dim=-1)
            start_loss = -torch.log(start_prob)
            end_loss = -torch.log(end_prob)
            span_loss = (start_loss + end_loss) / 2  # (bsz)
            span_loss.mul_(cls_label)  # (bsz)  => when sample label is 0, the span loss is not required.
            sum_loss = cls_loss + span_loss
            avg_loss = torch.mean(sum_loss)
            return avg_loss


class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate, warm_up_steps, decay_steps, min_lr_rate):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= (self.warm_up_steps + self.decay_steps):
            rate = (1.0 - ((self.optimizer_step - self.warm_up_steps) / self.decay_steps)) * self.init_rate
        else:
            rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()


class Main(object):
    def __init__(self, train_loader, valid_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = MyModel(pre_train_dir=args["pre_train_dir"], dropout_rate=args["dropout_rate"])

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args["weight_decay"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        self.optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args["init_lr"])
        self.schedule = WarmUp_LinearDecay(optimizer=self.optimizer, init_rate=args["init_lr"],
                                           warm_up_steps=args["warm_up_steps"],
                                           decay_steps=args["lr_decay_steps"], min_lr_rate=args["min_lr_rate"])
        self.model.to(device=device)

    def train(self):
        best_f = 0.0
        self.model.train()
        steps = 0
        while True:
            for item in self.train_loader:
                input_ids, input_mask, input_seg, cls_label, start_index, end_index, obj_mask, sub_mask, tim_mask, loc_mask = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["cls"], item["start_index"], item["end_index"], \
                    item["object_mask"], item["subject_mask"], item["time_mask"], item["location_mask"]
                self.optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids.to(device), input_mask=input_mask.to(device), input_seg=input_seg.to(device),
                    start_index=start_index.to(device), end_index=end_index.to(device), cls_label=cls_label.to(device),
                    object_mask=obj_mask.to(device), subject_mask=sub_mask.to(device), time_mask=tim_mask.to(device),
                    location_mask=loc_mask.to(device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args["clip_norm"])
                self.schedule.step()
                steps += 1
                if steps % self.args["print_interval"] == 0:
                    print("{} || [{}] || loss {:.3f}".format(
                        datetime.datetime.now(), steps, loss.item()
                    ))
                if steps % self.args["eval_interval"] == 0:
                    f, em = self.eval()
                    print("-*- eval F %.3f || EM %.3f -*-" % (f, em))
                    if f > best_f:
                        best_f = f
                        torch.save(self.model.state_dict(), f=self.args["save_path"])
                        print("current best model checkpoint has been saved successfully in ModelStorage")
                if steps >= self.args["max_steps"]:
                    break
            if steps >= self.args["max_steps"]:
                break

    def eval(self):
        self.model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for item in self.valid_loader:
                input_ids, input_mask, input_seg, label, context, context_range, arg_type = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["label"], item["context"], \
                    item["context_range"], item["type"]
                y_true.extend(label)
                cls, s_seq, e_seq = self.model(
                    input_ids=input_ids.to(device),
                    input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device)
                )
                cls = cls.cpu().numpy()
                s_seq = s_seq.cpu().numpy()
                e_seq = e_seq.cpu().numpy()
                for i in range(len(s_seq)):
                    y_pred.append(self.dynamic_search(cls[i], s_seq[i], e_seq[i], context[i], context_range[i], arg_type[i]))
        self.model.train()
        return self.calculate_f1(y_pred=y_pred, y_true=y_true)

    def dynamic_search(self, cls, s_seq, e_seq, context, context_range, arg_type):
        if cls[1] > cls[0]:
            max_score = 0.0
            dic = {"start": -1, "end": -1}
            t = context_range.split("-")
            start, end = int(t[0]), int(t[1])
            for i in range(start, end):
                for j in range(i, i + self.args["max_%s_len" % arg_type] if i + self.args["max_%s_len" % arg_type] <= end else end):
                    if s_seq[i] + e_seq[j] > max_score:
                        max_score = s_seq[i] + e_seq[j]
                        dic["start"], dic["end"] = i, j
            return context[dic["start"]-start:dic["end"]-start + 1]
        else:
            return ""

    @staticmethod
    def calculate_f1(y_pred, y_true):
        """
        :param y_pred: [n_samples]
        :param y_true: [n_samples]
        :return: 严格F1(exact match)和松弛F1(字符匹配率)
        """
        exact_match_cnt = 0
        char_match_cnt = 0
        char_pred_sum = char_true_sum = 1
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                exact_match_cnt += 1
            char_pred_sum += len(y_pred[i])
            char_true_sum += len(y_true[i])
            for j in y_pred[i]:
                if j in y_true[i]:
                    char_match_cnt += 1
        em = exact_match_cnt / len(y_true)
        precision_char = char_match_cnt / char_pred_sum
        recall_char = char_match_cnt / char_true_sum
        f1 = (2 * precision_char * recall_char) / (recall_char + precision_char)
        return (em + f1) / 2, em


if __name__ == "__main__":
    print("Hello RoBERTa Event Extraction.")
    device = "cuda:%s" % sys.argv[1][-1]
    args = {
        "init_lr": 2e-5,
        "batch_size": 12,
        "weight_decay": 0.01,
        "warm_up_steps": 2500,
        "lr_decay_steps": 6500,
        "max_steps": 12000,
        "min_lr_rate": 1e-9,
        "print_interval": 100,
        "eval_interval": 1000,
        "max_len": 512,
        "max_object_len": 25,  # 平均长度 7.22, 最大长度93
        "max_subject_len": 25,  # 平均长度10.0, 最大长度138
        "max_time_len": 20,  # 平均长度6.03, 最大长度22
        "max_location_len": 25,  # 平均长度3.79,最大长度41
        "save_path": "ModelStorage/argument.pth",
        "pre_train_dir": "/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/",
        "clip_norm": 0.25,
        "dropout_rate": 0.1
    }

    with open("DataSet/process.p", "rb") as f:
        x = pickle.load(f)

    tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")
    train_dataset = MyDataset(data=x["train_argument_items"], tokenizer=tokenizer, max_len=args["max_len"], special_query_token_map=x["argument_query_special_map_token"])
    valid_dataset = MyDataset(data=x["valid_argument_items"], tokenizer=tokenizer, max_len=args["max_len"], special_query_token_map=x["argument_query_special_map_token"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)

    m = Main(train_loader, valid_loader, args)
    m.train()


