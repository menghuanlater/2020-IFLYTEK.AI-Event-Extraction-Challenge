#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020-07-19
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage:
-----------------------------
    Description:
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
    def __init__(self, data, tokenizer: BertTokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        context, query, answers = item["context"], item["query"], item["answer"]
        # 首先编码input_ids ==> 分为Q和P两部分
        query_tokens = [i for i in query]
        context_tokens = [i for i in context]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len-1]
        c += ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_start = 2 + len(query_tokens)
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        start_seq_label, end_seq_label = [0] * self.max_len, [0] * self.max_len
        seq_mask = [0] * context_start + [1] * len(context_tokens) + [0] * (self.max_len - context_start - len(context_tokens))
        span_label = np.zeros(shape=(self.max_len, self.max_len), dtype=np.int32)
        triggers = []
        span_mask = np.zeros(shape=(self.max_len, self.max_len), dtype=np.float32)
        for item in answers:
            triggers.append(item["trigger"])
            start_seq_label[context_start + item["start"]] = 1
            end_seq_label[context_start + item["end"]] = 1
            span_label[context_start + item["start"], context_start + item["end"]] = 1
        for i in range(context_start, context_end):
            for j in range(i, context_end):
                span_mask[i, j] = 1.0
        return {
            "input_ids": torch.tensor(input_ids).long().to(),
            "input_seg": torch.tensor(input_seg).long(),
            "input_mask": torch.tensor(input_mask).float(),
            "context": context,
            "context_range": "%d-%d" % (context_start, context_end),  # 防止被转化成tensor
            "triggers": "&".join(triggers),
            "seq_mask": torch.tensor(seq_mask).float(),
            "start_seq_label": torch.tensor(start_seq_label).long(),
            "end_seq_label": torch.tensor(end_seq_label).long(),
            "span_label": torch.from_numpy(span_label).long(),
            "span_mask": torch.from_numpy(span_mask).float()
        }


class MyModel(torch.nn.Module):
    def __init__(self, pre_train_dir: str, dropout_rate: float, alpha, beta):
        """
        :param pre_train_dir: 预训练RoBERTa或者BERT文件夹
        :param dropout_rate: 随机失活率
        :param alpha: (start_loss + end_loss) / 2 的系数
        :param beta: span_loss 的系数 ==> 一般需要设置较低
        """
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_rate)
        )
        self.start_layer = torch.nn.Linear(in_features=1024, out_features=2)
        self.end_layer = torch.nn.Linear(in_features=1024, out_features=2)
        self.span1_layer = torch.nn.Linear(in_features=1024, out_features=1, bias=False)
        self.span2_layer = torch.nn.Linear(in_features=1024, out_features=1, bias=False)  # span1和span2是span_layer的拆解, 减少计算时的显存占用
        self.selfc = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).float().to(device), reduction="none")
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg, span_mask,
                start_seq_label=None, end_seq_label=None, span_label=None, seq_mask=None):
        bsz, seq_len = input_ids.size()[0], input_ids.size()[1]
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        start_logits = self.start_layer(encoder_rep)  # (bsz, seq, 2)
        end_logits = self.end_layer(encoder_rep)  # (bsz, seq, 2)
        span1_logits = self.span1_layer(encoder_rep)  # (bsz, seq, 1)
        span2_logits = self.span2_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        # 将两个span组合 => (bsz, seq, seq)
        span_logits = span1_logits.repeat(1, 1, seq_len) + span2_logits[:, None, :].repeat(1, seq_len, 1)
        start_prob_seq = torch.nn.functional.softmax(start_logits, dim=-1)  # (bsz, seq, 2)
        end_prob_seq = torch.nn.functional.softmax(end_logits, dim=-1)  # (bsz, seq, 2)
        # 使用span_mask
        span_logits.masked_fill_(span_mask == 0, -1e30)
        span_prob = torch.softmax(span_logits, dim=-1)  # (bsz, seq, seq)
        if start_seq_label is None or end_seq_label is None or span_label is None or seq_mask is None:
            return start_prob_seq, end_prob_seq, span_prob
        else:
            # 计算start和end的loss
            start_loss = self.selfc(input=start_logits.view(size=(-1, 2)), target=start_seq_label.view(size=(-1,)))
            end_loss = self.selfc(input=end_logits.view(size=(-1, 2)), target=end_seq_label.view(size=(-1,)))
            sum_loss = start_loss + end_loss
            sum_loss *= seq_mask.view(size=(-1,))
            avg_se_loss = self.alpha * torch.sum(sum_loss) / (torch.nonzero(seq_mask, as_tuple=False).size()[0])

            # 计算span loss
            span_loss = (-torch.log(span_prob + self.epsilon)) * span_label
            avg_span_loss = self.beta * torch.sum(span_loss) / (torch.nonzero(span_label, as_tuple=False).size()[0])
            return avg_se_loss + avg_span_loss


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
        self.model = MyModel(pre_train_dir=args["pre_train_dir"], dropout_rate=args["dropout_rate"], alpha=args["alpha"],
                             beta=args["beta"])

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
        self.model.to(device=args["device"])

    def train(self):
        best_em = 0.0
        self.model.train()
        steps = 0
        while True:
            for item in self.train_loader:
                input_ids, input_mask, input_seg, seq_mask, start_seq_label, end_seq_label, span_label, span_mask = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["seq_mask"], item["start_seq_label"], \
                    item["end_seq_label"], item["span_label"], item["span_mask"]
                self.optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids.to(self.args["device"]),
                    input_mask=input_mask.to(self.args["device"]),
                    input_seg=input_seg.to(self.args["device"]),
                    seq_mask=seq_mask.to(self.args["device"]),
                    start_seq_label=start_seq_label.to(self.args["device"]),
                    end_seq_label=end_seq_label.to(self.args["device"]),
                    span_label=span_label.to(self.args["device"]),
                    span_mask=span_mask.to(self.args["device"])
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
                    if em > best_em:
                        best_em = em
                        torch.save(self.model.state_dict(), f=self.args["save_path"])
                        print("current best model checkpoint has been saved successfully in ModelStorage")

    def eval(self):
        self.model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for item in self.valid_loader:
                input_ids, input_mask, input_seg, span_mask = item["input_ids"], item["input_mask"], item["input_seg"], item["span_mask"]
                y_true.extend(item["triggers"])
                s_seq, e_seq, p_seq = self.model(
                    input_ids=input_ids.to(self.args["device"]),
                    input_mask=input_mask.to(self.args["device"]),
                    input_seg=input_seg.to(self.args["device"]),
                    span_mask=span_mask.to(self.args["device"])
                )
                s_seq = s_seq.cpu().numpy()
                e_seq = e_seq.cpu().numpy()
                p_seq = p_seq.cpu().numpy()
                for i in range(len(s_seq)):
                    y_pred.append(self.dynamic_search(s_seq[i], e_seq[i], p_seq[i], item["context"][i], item["context_range"][i]))
        self.model.train()
        return self.calculate_f1(y_pred=y_pred, y_true=y_true)

    def dynamic_search(self, s_seq, e_seq, p_seq, context, context_range):
        ans_index = []
        t = context_range.split("-")
        c_start, c_end = int(t[0]), int(t[1])
        # 先找出所有被判别为开始和结束的位置索引
        i_start, i_end = [], []
        for i in range(c_start, c_end):
            if s_seq[i][1] > s_seq[i][0]:
                i_start.append(i)
            if e_seq[i][1] > e_seq[i][0]:
                i_end.append(i)
        # 然后遍历i_end
        cur_end = -1
        for e in i_end:
            s = []
            for i in i_start:
                if e >= i >= cur_end and (e - i) <= self.args["max_trigger_len"]:
                    s.append(i)
            max_s = 0.0
            t = None
            for i in s:
                if p_seq[i, e] > max_s:
                    t = (i, e)
                    max_s = p_seq[i, e]
            cur_end = e
            if t is not None:
                ans_index.append(t)
        out = []
        for item in ans_index:
            out.append(context[item[0] - c_start:item[1] - c_start + 1])
        return out

    @staticmethod
    def calculate_f1(y_pred, y_true):
        exact_match_cnt = 0
        exact_sum_cnt = 0
        char_match_cnt = 0
        char_pred_sum = char_true_sum = 0
        for i in range(len(y_true)):
            x = y_pred[i]
            y = y_true[i].split("&")
            exact_sum_cnt += len(y)
            for k in x:
                if k in y:
                    exact_match_cnt += 1
            x = "".join(x)
            y = "".join(y)
            char_pred_sum += len(x)
            char_true_sum += len(y)
            for k in x:
                if k in y:
                    char_match_cnt += 1
        em = exact_match_cnt / exact_sum_cnt
        precision_char = char_match_cnt / char_pred_sum
        recall_char = char_match_cnt / char_true_sum
        f1 = (2 * precision_char * recall_char) / (recall_char + precision_char)
        return (em + f1) / 2, em


if __name__ == "__main__":
    print("Hello RoBERTa Event Extraction.")
    device = "cuda:%s" % sys.argv[1][-1]
    args = {
        "device": device,
        "init_lr": 2e-5,
        "batch_size": 12,
        "weight_decay": 0.01,
        "warm_up_steps": 1000,
        "lr_decay_steps": 4000,
        "max_steps": 5000,
        "min_lr_rate": 1e-9,
        "print_interval": 100,
        "eval_interval": 500,
        "max_len": 512,
        "max_trigger_len": 6,
        "save_path": "ModelStorage/dominant_trigger.pth",
        "pre_train_dir": "/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/",
        "clip_norm": 0.25,
        "dropout_rate": 0.1,
        "alpha": 1.0,
        "beta": 1.0,
    }

    with open("DataSet/process.p", "rb") as f:
        x = pickle.load(f)

    tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")
    train_dataset = MyDataset(data=x["train_dominant_trigger_items"], tokenizer=tokenizer, max_len=args["max_len"])
    valid_dataset = MyDataset(data=x["valid_dominant_trigger_items"], tokenizer=tokenizer, max_len=args["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)

    m = Main(train_loader, valid_loader, args)
    m.train()

