<div style="font-size: 1.5rem;">
  <a href="./README.md">中文</a> |
  <a href="./docs/readme_en.md">English</a>
</div>
</br>

<h1 align="center">FastTextClassification</h1>
<div align="center">
  <a href="https://github.com/fast-llm/FastTextClassification">
    <img src="https://pic4.zhimg.com/80/v2-f63d74cf9859eea57b0a78c9da00c9f3_720w.webp" alt="Logo" height="210">
  </a>
  <p align="center">
    <h3>Fast text classification for you, Start your NLP journey</h3>
      <a href="https://github.com/fast-llm/FastTextClassification/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/fast-llm/FastTextClassification" />
      </a>
      <a href="https://github.com/fast-llm/FastTextClassification/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/fast-llm/FastTextClassification?color=0088ff" />
      </a>
      <a href="https://github.com/fast-llm/FastTextClassification/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/fast-llm/FastTextClassification?color=0088ff" />
      </a>
      <a href="https://github.com/fast-llm/FastTextClassification/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/fast-llm/FastTextClassification?color=0088ff" />
      <a href="https://github.com/fast-llm/FastTextClassification/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/fast-llm/FastTextClassification?color=ccf" />
      </a>
      <br/>
      <em>开源实现 / 简单 / 全面 / 实践 </em>
      <br/>
    </p>
 </p>
</div>

> **功能免费，代码开源，大家放心使用，欢迎贡献！**

- [💥最新讯息](#最新讯息)
- [⛏️使用步骤](#使用步骤)
- [📄运行示例](#运行示例)
- [📄结果展示](#结果展示)
- [🛠️常见报错](#常见报错)
- [💐参考资料&致谢](#参考资料&致谢)

## 最新讯息

- 2024/05/06: ernie支持多卡、slurm等分布式训练
- 2024/04/30: clone原项目进行再次开发

- 2023/03/23：FastTextClassification V0.0.1版正式开源，版本特性：
  - 支持中英双语的文本分类
  - 支持多种文本分类模型：传统机器学习浅层模型、深度学习模型和transformers类模型
  - 支持多标签文本分类
  - 支持多种embedding方式：inner/outer/random


### 开发计划

本项目的开发宗旨，打造全网最全面和最实用的文本分类项目和教程。如果有机会，未来希望可以做成开箱即用的文本分类工具，文本分类任务非常特殊，大部分情况下被认为是简单且基础的，然而却很难找到比较通用的文本分类工具，往往都是针对具体任务进行训练和部署。在NLP逐渐趋于大一统的今天，这一点非常不优雅，而且浪费资源。：***Fast text classification for you, Start your NLP journey!\***

**简要的开发计划**：

1. 【P3】支持中英双语的文本分类：100%，也欢迎支持其他语种
2. 【P0】支持多种文本分类模型：基本完成，欢迎补充
	1. 浅层文本分类模型：done
	2. 【P1】DNN类模型：已支持常见模型
	3. 【P0】transformer类模型：Bert/ERNIE等
	4. 【P0】prompt learning for Text Classification：TODO
	5. 【P0】ChatGPT for Text Classification：TODO
3. 【P1】支持多标签文本分类：
	1. 多种多标签分类loss：done，如有遗漏，欢迎补充
	2. 复杂的多标签分类：比如层次化等，TODO
4. 【P0】支持不同的文本分类数据集/任务：文本分类任务又多又散，这是好事儿也是坏事儿。欢迎基于本项目报告各种数据集上的效果
5. 【P4】支持简明易用的文本分类API：终极目标为实现一个足够通用和强大的文本分类模型，并实现自然语言交互的文本分类接口text_cls(text, candidate_labels)->label，给定文本和候选类别(有默认值)，输出文本所属的类别；同时支持可无成本或尽可能小的成本向特定领域泛化
6. 多GPU和集群训练: TODO


## 使用步骤

1.克隆本项目

`git clone https://github.com/fast-llm/FastTextClassification.git`

2.数据集下载和预处理

请自行下载数据集，将其放到data目录下，数据统一处理成text+label格式，以\t或逗号分隔。有空我再来补一个自动化脚本，暂时请自行处理或者参考preprocessing.py。

最好将数据统一放到data目录下，比如data/dbpedia，然后分3个子目录，input存放原始数据集（你下载的数据集），data存放预处理后的格式化的数据集（text-label格式），saved_dict存放训练结果（模型和日志等）。

3.运行示例

经过测试的开发环境如下，仅供参考，差不多的环境应该都可以运行。

- python：3.6/3.7
- torch：1.6.0
- transformers：4.18.0
- torchtext：0.7.0
- scikit-learn： 0.24.2
- tensorboardX：2.6
- nltk：3.6.7
- numpy：1.18.5
- pandas：1.1.5



根据自己的需要选择模块运行，详见下一节。

` python run.py`

## 运行示例

1.运行DNN/transformers类模型做文本分类

` python run.py`

2.运行传统浅层机器学习模型做文本分类

`python run_shallow.py`

3.运行DNN/transformers类模型做多标签文本分类

`python run_multi_label.py`



下表是直接运行demo的参考结果：

运行环境：python3.6 + T4

| demo               | 数据集      | 示例模型 | Acc    | 耗时      | 备注               |
| ------------------ | ----------- | -------- | ------ | --------- | ------------------ |
| run.py             | THUCNews/cn | TextCNN  | 89.94% | ~2mins    |                    |
| run_multi_label.py | rcv1/en     | bert     | 61.04% | ~40mins   | 其他指标见运行结果 |
| run_shallow.py     | THUCNews/cn | NB       | 89.44% | 105.34 ms |                    |

## 结果展示：持续更新中

笔者提供了从浅到深再到多标签的详细实验结果，可供大家参考。但受限于时间和算力，很多实验可能未达到最优，望知悉！因此，非常欢迎大家积极贡献，补充相关实验、代码和新的模型等等，一起建设FastTextClassification。

暂时只提供部分汇总的结果，详细的实验结果及参数等我有空再补，比较多，需要一些时间整理。
### 1.传统浅层文本分类模型

| Data        | Model                    | tokenizer | 最小词长 | Min_df | ngram | binary | Use_idf | Test acc | 备注                                                         |
| ----------- | ------------------------ | --------- | -------- | ------ | ----- | ------ | ------- | -------- | ------------------------------------------------------------ |
| THUCNews/cn | LR                       | lcut      | 1        | 2      | (1,1) | False  | True    | 90.61%   | C=1.0, max_iter=1000  词表61549；  train score:  94.22%  valid score:  89.84%  test score: 90.61%  training time:  175070.97 ms |
|             | MultinomialNB(alpha=0.3) | lcut      | 1        | 2      | (1,1) | False  | True    | 89.86%   | 词表61549；  training time: 94.18ms                          |
|             | ComplementNB(alpha=0.8)  | lcut      | 1        | 2      | (1,1) | False  | True    | 89.88%   | 词表61549；  training time: 98.31ms                          |
|             | SVC(C=1.0)               | lcut      | 1        | 2      | (1,1) | False  | True    | 81.49%   | 词表61549；  维度200  training time:  7351155.59 ms  train score:  85.95%  valid score:  80.07%  test score: 81.49% |
|             | DT                       | lcut      | 1        | 2      | (1,1) | False  | True    | 71.19%   | max_depth=None     training time:  149216.53 ms  train score:  99.97%  valid score:  70.57%  test score: 71.19% |
|             | xgboost                  | lcut      | 1        | 2      | (1,1) | False  | True    | 90.08%   | XGBClassifier(n_estimators=2000,eta=0.3,gamma=0.1,max_depth=6,subsample=1,colsample_bytree=0.8,  nthread=10)  training time:  1551260.28 ms  train score:  99.00%  valid score:  89.34%  test score: 90.08% |
|             | KNN                      | lcut      | 1        | 2      | (1,1) | False  | True    | 85.17%   | k=10  training time:  21.24 ms  train score:  89.05%  valid score:  84.53%  test score: 85.17% |
|             |                          |           |          |        |       |        |         |          |                                                              |
| dbpedia/en  | LR                       | None      | 2        | 2      | (1,1) | False  | True    | 98.26%   | C=1.0, max_iter=100  词表237777  training time:  220177.59 ms  train score:  98.85%  valid score:  98.19%  test score: 98.26% |
|             | MultinomialNB(alpha=1.0) | None      | 2        | 2      | (1,1) | False  | True    | 95.35%   | training time:  786.24 ms  train score:  96.36%  valid score:  95.34%  test score: 95.35% |
|             | ComplementNB(alpha=1.0)  | None      | 2        | 2      | (1,1) | False  | True    | 93.73%   | training time:  805.69 ms  train score:  95.30%  valid score:  93.79%  test score: 93.73% |
|             | SVC(C=1.0)               | None      | 2        | 2      | (1,1) | False  | True    | 94.67%   | 维度200；  max_iter=100     training time:  144163.81 ms  train score:  94.75%  valid score:  94.59%  test score: 94.67%  注意：SVM的计算和存储成本正比于样本数的平方； |
|             | DT                       | None      | 2        | 2      | (1,1) | False  | True    | 92.41%   | max_depth=100,  min_samples_leaf=5     training  time: 639744.56 ms  train  score: 95.79%  valid  score: 92.43%  test  score: 92.41% |
|             | xgboost                  | None      | 2        | 2      | (1,1) | False  | True    | 97.99%   | XGBClassifier(n_estimators=200,eta=0.3,gamma=0.1,max_depth=6,subsample=1,colsample_bytree=0.8,  nthread=10,reg_alpha=0,reg_lambda=1)     training time:  1838434.42 ms  train score:  99.35%  valid score:  97.96%  test score: 97.99% |
|             | KNN                      | None      | 2        | 2      | (1,1) | False  | True    | 80.05%   | k=10  training time:  137.72 ms  train score:  84.66%  valid score:  80.20%  test score: 80.05% |
|             |                          |           |          |        |       |        |         |          |                                                              |

###  2.深度学习文本分类模型

| Data        | Model       | Embed | Bz   | Lr   | epochs | acc    | 备注              |
| ----------- | ----------- | ----- | ---- | ---- | ------ | ------ | ----------------- |
| THUCNews/cn | TextCNN     | outer | 128  | 1e-3 | 3/20   | 90.45% |                   |
|             | TextRNN     | -     | -    | 1e-3 | 5/10   | 90.38% |                   |
|             | TextRNN_Att |       |      | 1e-3 | 2/10   | 90.55% |                   |
|             | TextRCNN    |       |      | 1e-3 | 3/10   | 91.01% |                   |
|             | DPCNN       |       |      | 1e-3 | 3/20   | 90.12% |                   |
|             | FastText    |       |      | 1e-3 | 5/20   | 90.48% |                   |
|             | bert        | inner |      | 5e-5 | 2/3    | 94.10% | bert-base-chinese |
|             | ERNIE       | inner |      | 5e-5 | 3/3    | 94.58% | ernie-3.0-base-zh |
|             | bert_CNN    |       |      | -    | 3/3    | 94.14% |                   |
|             | bert_RNN    |       |      | -    | 3/3    | 93.92% |                   |
|             | bert_RNN    |       |      | -    | 3/3    | 94.45% |                   |
|             | bert_RCNN   |       |      | -    | 3/3    | 94.32% |                   |
|             | bert_DPCNN  |       |      | -    | 3/3    | 94.17% |                   |
|             |             |       |      |      |        |        |                   |
| dbpedia/en  | TextCNN     | outer | 128  | 5e-5 | 9/20   | 98.35% | glove             |
|             | TextRNN     | -     | -    | -    | 6/10   | 97.97% |                   |
|             | TextRNN_Att |       |      | -    | 4/10   | 97.80% |                   |
|             | TextRCNN    |       |      | -    | 3/10   | 97.71% |                   |
|             | DPCNN       |       |      | -    | 3/20   | 97.86% |                   |
|             | FastText    |       |      | -    | 10/20  | 97.84% |                   |
|             | bert        | inner |      | 5e-5 | 2/3    | 97.78% | bert-base-uncased |
|             | ERNIE       |       |      |      | 2/10   | 97.75% | ernie-2.0-base-en |
|             | bert_CNN    |       |      | -    | 2/3    | 97.91% |                   |
|             | bert_RNN    |       |      | -    | 2/3    | 97.87% |                   |
|             | bert_RCNN   |       |      | -    | 2/3    | 98.04% |                   |
|             | bert_DPCNN  |       |      | -    | 2/3    | 97.95% |                   |
|             |  gpt        |       |      |      | 3/3    | 97.03  |                   |
|             |  gpt2       |       |      |      | 3/3    | 97.00  |                   |
|             |  T5         |       |      |      | 3/3    | 96.57  |                   |
|             |             |       |      |      |        |        |                   |

### 3.多标签文本分类

| Data    | Model       | 分层 | 样本数 | Embed | loss                    | Bz   | Lr   | epochs | Test acc  (绝对匹配率） | Micro-F1 | Macro-F1 | 备注                                    |
| ------- | ----------- | ---- | ------ | ----- | ----------------------- | ---- | ---- | ------ | ----------------------- | -------- | -------- | --------------------------------------- |
| Rcv1/en | TextCNN     | -    | all    | outer | multi_label_circle_loss | 128  | 1e-3 | 9/20   | 51.02%                  | 0.7904   | 0.4515   | eval_activate = None  cls_threshold = 0 |
|         | TextRNN     |      |        | -     |                         | -    | -    | 13/20  | 54.00%                  | 0.7950   | 0.4358   |                                         |
|         | TextRNN_Att |      |        |       |                         |      | -    | 11/20  | 53.97%                  | 0.8011   | 0.4538   |                                         |
|         | TextRCNN    |      |        |       |                         |      | -    | 10/20  | 53.62%                  | 0.8111   | 0.4900   |                                         |
|         | DPCNN       |      |        |       |                         |      | -    | 10/20  | 51.66%                  | 0.7890   | 0.4111   |                                         |
|         | FastText    |      |        |       |                         |      | -    | 12/20  | 51.31%                  | 0.7936   | 0.4728   |                                         |
|         | bert        |      | all    | inner | -                       | 128  | 2e-5 | 20/20  | 61.04%                  | 0.8454   | 0.5729   | bert-base-cased                         |
|         | ERNIE       |      | all    | inner | -                       | 128  | 2e-5 | 20/20  | 61.67%                  | 0.8486   | 0.5861   | ernie-2.0-base-en                       |
|         | Bert_CNN    |      | all    | inner | -                       | 128  | 2e-5 | 12/20  | 58.31%                  | 0.8364   | 0.5736   | 同bert配置                              |
|         | Bert_RNN    |      | all    | inner | -                       | 128  | 2e-5 | 17/20  | 60.48%                  | 0.8371   | 0.5640   |                                         |
|         | Bert_RCNN   |      | all    | inner | -                       | 128  | 2e-5 | 15/20  | 60.54%                  | 0.8457   | 0.5969   |                                         |
|         | Bert_DPCNN  |      | all    | inner | -                       | 128  | 2e-5 | 13/20  | 56.52%                  | 0.8082   | 0.4273   |                                         |
|         |             |      |        |       |                         |      |      |        |                         |          |          |                                         |


## 常见报错



## 参考资料&致谢

A Survey on Text Classification: From Shallow to Deep Learning：https://arxiv.org/pdf/2008.00364.pdf?utm_source=summari

Deep Learning--based Text Classification: A Comprehensive Review：https://arxiv.org/pdf/2004.03705.pdf

https://github.com/649453932/Chinese-Text-Classification-Pytorch

https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

https://github.com/facebookresearch/fastText

https://github.com/brightmart/text_classification

https://github.com/kk7nc/Text_Classification

https://github.com/Tencent/NeuralNLP-NeuralClassifier

https://github.com/vandit15/Class-balanced-loss-pytorch

https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics


## 赞助我们

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=fast-llm/FastTextClassification&type=Date)](https://star-history.com/#fast-llm/FastTextClassification&Date)


## 贡献者

<a href="https://github.com/fast-llm/FastTextClassification/graphs/contributors">
   <img src="https://contrib.rocks/image?repo=fast-llm/FastTextClassification" />
</a>