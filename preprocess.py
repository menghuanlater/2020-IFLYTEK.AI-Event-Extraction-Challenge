#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020-07-23
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage: 
-----------------------------
    Description: 数据预处理
    返回数据字典格式:
    1. 主触发词识别模型的标注数据: {
        "id": str, "context": str, "query": "找出事件中的触发词", "answer": [{"trigger": str, "start": int, "end": int}...],
    }  => 一个新闻一个数据项

    2. 辅助触发词识别模型的标注数据: {
        "id": str, "context": str, "query": "找出事件中的触发词", "answer": {"trigger": str, "start": int, "end": int}
    }  => 一个触发词一个数据项

    3. 论元抽取模型的标注数据: {
        "id": str, "context": str, "query": "处于位置&i&和位置-j-之间的触发词*s*的x为?", "answer": {"argument": str, "is_exist": bool, "start": int, "end": int},
        "type": str
    }  => 一个论元一个数据项 ==> 一个事件四个数据项(主体 客体 时间 地点)
-----------------------------
"""
import pickle
import csv
import jieba
from random import shuffle

valid_dominant_trigger_number = 600
valid_argument_number = 1000

train_file = open("DataSet/train/train.csv", "r", encoding="UTF-8")  # 训练文件
test_file = open("DataSet/test/test.csv", "r", encoding="UTF-8")  # 测试文件
sample_file = open("DataSet/test/sample.csv", "r", encoding="UTF-8")

train_reader = csv.reader(train_file)
test_reader = csv.reader(test_file)
sample_reader = csv.reader(sample_file)
next(train_reader)
next(test_reader)
next(sample_reader)

output = {
    "train_dominant_trigger_items": None,
    "valid_dominant_trigger_items": None,
    "train_argument_items": [],
    "valid_argument_items": [],
    "train_aux_trigger_items": None,
    "test_items": [],  # {"id", "context"},
    "argument_query_special_map_token": {
        "&": "[unused1]", "-": "[unused2]", "*": "[unused3]"
    }
}

all_triggers = dict()
object_arguments = {"exist": [], "not_exist": []}
time_arguments = {"exist": [], "not_exist": []}
subject_arguments = {"exist": [], "not_exist": []}
location_arguments = {"exist": [], "not_exist": []}

for item in train_reader:
    if item[0] not in all_triggers.keys():
        all_triggers[item[0]] = {
            "id": item[0], "context": item[1], "answer": list(), "query": "找出事件中的触发词"
        }
    obj = all_triggers[item[0]]["answer"]
    _context, _trigger, _object, _subject, _time, _location = \
        item[1].replace("－", "-").replace("～", "~"), item[2], item[3].replace("－", "-"), item[4].replace("－", "-"), \
        item[5].replace("－", "-"), item[6].replace("－", "-")
    # 特殊化处理(仅仅训练集存在这种情况->将所有的变种0~9进行替换)
    for i in range(10):
        r_c = chr(65296 + i)
        _context = _context.replace(r_c, "%d" % i)
        _trigger = _trigger.replace(r_c, "%d" % i)
        _object = _object.replace(r_c, "%d" % i)
        _subject = _subject.replace(r_c, "%d" % i)
        _time = _time.replace(r_c, "%d" % i)
        _location = _location.replace(r_c, "%d" % i)
    trigger_index = len(obj)
    # 首先处理triggers
    x = list(jieba.tokenize(_context))  # 切词带索引
    y = jieba.lcut(_context)  # 单纯的切词序列
    assert len(x) == len(y)

    overlap_flag = False
    __context = ""
    overlap_index = -1

    for i in range(trigger_index):
        if obj[trigger_index-1-i]["trigger"] == _trigger or _trigger in obj[trigger_index-1-i]["trigger"]:
            overlap_flag = True
            x = list(jieba.tokenize(_context[obj[trigger_index-1-i]["end"]+1:]))
            y = jieba.lcut(_context[obj[trigger_index-1-i]["end"]+1:])
            __context = _context[obj[trigger_index-1-i]["end"]+1:]
            overlap_index = trigger_index-1-i
            break

    # 需要检测前序触发词是否已经出现,若出现必须从那个词的end开始重新处理x和y
    if _trigger in y:
        index = y.index(_trigger)
        obj.append({
            "trigger": _trigger, "start": x[index][1], "end": x[index][2] - 1
        })
    else:
        if overlap_flag:
            if _trigger in __context:
                index = __context.index(_trigger)
            else:
                index = obj[overlap_index]["start"]
        else:
            index = _context.index(_trigger)
        obj.append({
            "trigger": _trigger, "start": index, "end": index + len(_trigger) - 1
        })

    # 处理论元
    obj_tmp = {"type": "object", "id": item[0], "context": item[1], "query": "处于位置&%d&和位置-%d-之间的触发词*%s*的主体为?" % (obj[-1]["start"], obj[-1]["end"], _trigger)}
    sub_tmp = {"type": "subject", "id": item[0], "context": item[1], "query": "处于位置&%d&和位置-%d-之间的触发词*%s*的客体为?" % (obj[-1]["start"], obj[-1]["end"], _trigger)}
    tim_tmp = {"type": "time", "id": item[0], "context": item[1], "query": "处于位置&%d&和位置-%d-之间的触发词*%s*的时间为?" % (obj[-1]["start"], obj[-1]["end"], _trigger)}
    loc_tmp = {"type": "location", "id": item[0], "context": item[1], "query": "处于位置&%d&和位置-%d-之间的触发词*%s*的地点为?" % (obj[-1]["start"], obj[-1]["end"], _trigger)}
    if _object == "":
        obj_tmp["answer"] = {"is_exist": 0, "start": -1, "end": -1, "argument": _object}
    else:
        index = _context.index(_object)
        obj_tmp["answer"] = {"is_exist": 1, "start": index, "end": index + len(_object) - 1, "argument": _object}
    if _subject == "":
        sub_tmp["answer"] = {"is_exist": 0, "start": -1, "end": -1, "argument": _subject}
    else:
        index = _context.index(_subject)
        sub_tmp["answer"] = {"is_exist": 1, "start": index, "end": index + len(_subject) - 1, "argument": _subject}
    if _time == "":
        tim_tmp["answer"] = {"is_exist": 0, "start": -1, "end": -1, "argument": _time}
    else:
        index = _context.index(_time)
        tim_tmp["answer"] = {"is_exist": 1, "start": index, "end": index + len(_time) - 1, "argument": _time}
    if _location == "":
        loc_tmp["answer"] = {"is_exist": 0, "start": -1, "end": -1, "argument": _location}
    else:
        try:
            index = _context.index(_location)
            loc_tmp["answer"] = {"is_exist": 1, "start": index, "end": index + len(_location) - 1, "argument": _location}
        except ValueError:
            if _location in ["福建内", "湖北中", "阿拉善边", "山东后", "宁都上", "庐山上"]:
                _location = _location[:-1]
                index = _context.index(_location)
                loc_tmp["answer"] = {"is_exist": 1, "start": index, "end": index + len(_location) - 1, "argument": _location}
            else:
                loc_tmp["answer"] = {"is_exist": 0, "start": -1, "end": -1, "argument": ""}
    if obj_tmp["answer"]["is_exist"] == 1:
        object_arguments["exist"].append(obj_tmp)
    else:
        object_arguments["not_exist"].append(obj_tmp)
    if sub_tmp["answer"]["is_exist"] == 1:
        subject_arguments["exist"].append(sub_tmp)
    else:
        subject_arguments["not_exist"].append(sub_tmp)
    if tim_tmp["answer"]["is_exist"] == 1:
        time_arguments["exist"].append(tim_tmp)
    else:
        time_arguments["not_exist"].append(tim_tmp)
    if loc_tmp["answer"]["is_exist"] == 1:
        location_arguments["exist"].append(loc_tmp)
    else:
        location_arguments["not_exist"].append(loc_tmp)

# step1: 整理划分触发词识别模型
dominant_items = []
aux_items = []
for key in all_triggers.keys():
    dominant_items.append(all_triggers[key])
    for item in all_triggers[key]["answer"]:
        aux_items.append({
            "id": all_triggers[key]["id"],
            "context": all_triggers[key]["context"],
            "query": all_triggers[key]["query"],
            "answer": item
        })
for i in range(3):
    shuffle(dominant_items)
    shuffle(aux_items)
    shuffle(object_arguments["exist"])
    shuffle(object_arguments["not_exist"])
    shuffle(subject_arguments["exist"])
    shuffle(subject_arguments["not_exist"])
    shuffle(time_arguments["exist"])
    shuffle(time_arguments["not_exist"])
    shuffle(location_arguments["exist"])
    shuffle(location_arguments["not_exist"])
output["train_dominant_trigger_items"] = dominant_items[valid_dominant_trigger_number:]
output["valid_dominant_trigger_items"] = dominant_items[:valid_dominant_trigger_number]
output["train_aux_trigger_items"] = aux_items

# step2: 整理划分各论元数据
x = int((len(object_arguments["exist"]) / 8000) * valid_argument_number)
y = valid_argument_number - x
output["train_argument_items"].extend(object_arguments["exist"][x:] + object_arguments["not_exist"][y:])
output["valid_argument_items"].extend(object_arguments["exist"][:x] + object_arguments["not_exist"][:y])

x = int((len(subject_arguments["exist"]) / 8000) * valid_argument_number)
y = valid_argument_number - x
output["train_argument_items"].extend(subject_arguments["exist"][x:] + subject_arguments["not_exist"][y:])
output["valid_argument_items"].extend(subject_arguments["exist"][:x] + subject_arguments["not_exist"][:y])

x = int((len(time_arguments["exist"]) / 8000) * valid_argument_number)
y = valid_argument_number - x
output["train_argument_items"].extend(time_arguments["exist"][x:] + time_arguments["not_exist"][y:])
output["valid_argument_items"].extend(time_arguments["exist"][:x] + time_arguments["not_exist"][:y])

x = int((len(location_arguments["exist"]) / 8000) * valid_argument_number)
y = valid_argument_number - x
output["train_argument_items"].extend(location_arguments["exist"][x:] + location_arguments["not_exist"][y:])
output["valid_argument_items"].extend(location_arguments["exist"][:x] + location_arguments["not_exist"][:y])

for i in range(10):
    shuffle(output["train_argument_items"])

tmp = dict()
for item in sample_reader:
    if item[0] not in tmp.keys():
        tmp[item[0]] = 1
    else:
        tmp[item[0]] += 1

for item in test_reader:
    output["test_items"].append(
        {"id": item[0], "context": item[1], "n_triggers": tmp[item[0]]}
    )

with open("DataSet/process.p", "wb") as f:
    pickle.dump(output, f)