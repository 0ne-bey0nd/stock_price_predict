# 股票价格预测 v0.1.1

---


## 支持模型

- []() LSTM

## 架构

- 数据过程
  - 原始数据爬取
  - 原始数据处理
  - 数据接口
- 模型训练过程
  - 数据集准备过程
    - 数据自动标签
    - 数据序列建模
    - 数据拆分和预处理
- 模型预测过程
  - 完善中


## 问题建模

（待完善）

### Sequence-to-One（Seq2One）

输入是序列，输出是单值：这种模型的输入是一个变长的序列，但输出是一个固定的单值或单个向量。

输入的序列即股票不同时间的价格，输出即预测目标，可以是下一个时间点股票的涨跌性质（分类），也可以是下一个时间点股票的价格（回归）


## 相关工作引用

- [会有的]()

## 更新信息

---
### v0.1.1

- refactor: change the data set label and construct features part, bring the feature scale to the predict phase

### problems

- the model only takes the market price info(open, high, low, close and volume) as input, and at v0.1.1 when every bug in code seems to be fixed, the accuracy is terribly low( nearly 49.58% and lower) in the task that takes 7 days features to predict the close price of the next unknown day. From the view of data mining, all these may illustrate that it's bullshit to stare only at the price line if you want to have a good performance at your investment.

### v0.1.2

- 

---