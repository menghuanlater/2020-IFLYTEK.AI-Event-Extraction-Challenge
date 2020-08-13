step1: run preprocess.py
step2: {
    run auxiliary_trigger.py;
    run dominant_trigger.py;
    run argument.py;
}
step3: run joint_predict.py

核心思路: 以RoBERTa_Large作为Context Encoder, 将问题拆解为触发词识别和触发词对应四论元检测
触发词识别和论元识别基于MRC(Span Extraction类机器阅读理解)思路实现

参考论文: ACL2020 《A Unified MRC Framework for Named Entity Recognition》

single model score: 0.78