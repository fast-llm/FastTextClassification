<div style="font-size: 1.5rem;">
   <a href="./README.md">Chinese</a> |
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

> **The function is free, the code is open source, everyone can use it with confidence, welcome to contribute! **


- [💥Latest News](#Latest News)
- [💫Development plan](#Development plan)
- [⛏️Steps to use](#Steps to use)
- [📄Run Example](#Run Example)
- [📄Result Display](#Result Display)
- [🛠️Common error report](#Common error report)
- [💐References & Acknowledgments] (#References & Acknowledgments)
- [🌟Sponsor us](#Sponsor us)
- [🌈Starchart] (#Starchart)
- [🏆Contributors] (#Contributors)



## Latest News

- 2023/03/23: FastTextClassification V0.0.1 is officially open source, version features:
	- Support bilingual text classification in Chinese and English
	- Support multiple text classification models: traditional machine learning shallow model, deep learning model and transformers class model
	- Support multi-label text classification
	- Support multiple embedding methods: inner/outer/random



## Steps for usage

1. Clone this project

`git clone https://github.com/fast-llm/FastTextClassification.git`

2. Dataset download and preprocessing

I will add another automation script when I have time, please handle it by yourself or refer to preprocessing.py for the time being.

It is best to put the data in the data directory, such as data/dbpedia, and then divide it into 3 subdirectories. Input stores the original data set (the data set you downloaded), and data stores the preprocessed formatted data set (text-label format), saved_dict stores the training results (model and log, etc.).

3. Run the example

The tested development environment is as follows, for reference only, almost all environments should be able to run.

- python: 3.6/3.7
- torch: 1.6.0
- transformers: 4.18.0
- torchtext: 0.7.0
- scikit-learn: 0.24.2
-tensorboardX:2.6
-nltk:3.6.7
- numpy: 1.18.5
- pandas: 1.1.5



Select the module to run according to your own needs, see the next section for details.

`python run.py`

## Run the example

1. Run the DNN/transformers class model for text classification

`python run.py`

2. Run the traditional shallow machine learning model for text classification

`python run_shallow.py`

3. Run the DNN/transformers class model to do multi-label text classification

`python run_multi_label.py`



The following table is the reference result of running the demo directly:

Operating environment: python3.6 + T4

| demo               | Dataset     | Example Model | Acc    | Time Consumption | Remarks                                  |
| ------------------ | ----------- | ------------- | ------ | ---------------- | ---------------------------------------- |
| run.py             | THUCNews/cn | TextCNN       | 89.94% | ~2mins           |                                          |
| run_multi_label.py | rcv1/en     | bert          | 61.04% | ~40mins          | See running results for other indicators |
| run_shallow.py     | THUCNews/cn | NB            | 89.44% | 105.34 ms        |                                          |

## Result display: to be continued

The author provides detailed experimental results from shallow to deep to multi-label for your reference. However, due to limited time and computing power, many experiments may not be optimal, please know! Therefore, everyone is very welcome to actively contribute, supplement related experiments, codes, new models, etc., and build FastTextClassification together.

For the time being, only part of the summary results are provided. I will add detailed experimental results and parameters when I have time. There are many, and it will take some time to sort them out.

| Data     | Model                    | tokenizer | Min_word_len | Min_df | ngram | binary | Use_idf | Test acc | 备注                                                         |
| -------- | ------------------------ | --------- | ------------ | ------ | ----- | ------ | ------- | -------- | ------------------------------------------------------------ |
| THUCNews | LR                       | lcut      | 1            | 2      | (1,1) | False  | True    | 90.61%   | C=1.0, max_iter=1000  词表61549；  train score:  94.22%  valid score:  89.84%  test score: 90.61%  training time:  175070.97 ms |
|          | MultinomialNB(alpha=0.3) | lcut      | 1            | 2      | (1,1) | False  | True    | 89.86%   | 词表61549；  training time: 94.18ms                          |
|          | ComplementNB(alpha=0.8)  | lcut      | 1            | 2      | (1,1) | False  | True    | 89.88%   | 词表61549；  training time: 98.31ms                          |
|          | SVC(C=1.0)               | lcut      | 1            | 2      | (1,1) | False  | True    | 81.49%   | 词表61549；  维度200  training time:  7351155.59 ms  train score:  85.95%  valid score:  80.07%  test score: 81.49% |
|          | DT                       | lcut      | 1            | 2      | (1,1) | False  | True    | 71.19%   | max_depth=None     training time:  149216.53 ms  train score:  99.97%  valid score:  70.57%  test score: 71.19% |
|          | xgboost                  | lcut      | 1            | 2      | (1,1) | False  | True    | 90.08%   | XGBClassifier(n_estimators=2000,eta=0.3,gamma=0.1,max_depth=6,subsample=1,colsample_bytree=0.8,  nthread=10)  training time:  1551260.28 ms  train score:  99.00%  valid score:  89.34%  test score: 90.08% |
|          | KNN                      | lcut      | 1            | 2      | (1,1) | False  | True    | 83.34%   | k=5  training time:  22.14 ms  train score:  89.57%  valid score:  82.69%  test score: 83.34% |
|          |                          |           |              |        |       |        |         |          |                                                              |

 

## Common errors



## References & Acknowledgments

A Survey on Text Classification: From Shallow to Deep Learning: https://arxiv.org/pdf/2008.00364.pdf?utm_source=summari

Deep Learning--based Text Classification: A Comprehensive Review: https://arxiv.org/pdf/2004.03705.pdf

https://github.com/649453932/Chinese-Text-Classification-Pytorch

https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

https://github.com/facebookresearch/fastText

https://github.com/brightmart/text_classification

https://github.com/kk7nc/Text_Classification

https://github.com/Tencent/NeuralNLP-NeuralClassifier

https://github.com/vandit15/Class-balanced-loss-pytorch

https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics


## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=fast-llm/FastTextClassification&type=Date)](https://star-history.com/#kaixindelele/ChatPaper&Date)

## Contributors

<a href="https://github.com/fast-llm/FastTextClassification/graphs/contributors">
   <img src="https://contrib.rocks/image?repo=fast-llm/FastTextClassification" />
</a>