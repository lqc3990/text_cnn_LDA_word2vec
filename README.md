# text_cnn_LDA_word2vec
---
简介：
---

对用户的餐饮评论文本进行情感分析，评价用户的消费感受，可帮助商家了解用户感受，也能够辅助实现用户推荐系统。
针对传统的情感分析方法提取特征单一、文本理解片面和文本表示不充分的缺点，本项目提出结合Word2vec和隐含狄利克雷分布提取多层次特征，同时挖掘文本的上下文和语义特征。
同时，利用卷积神经网络（CNN）能够捕捉局部信息的特点，对提取到的特征矩阵进一步挖掘，最后在SoftMax分类器上做情感倾向分类。
实验证明，该方法在准确率和F值的度量上，较单一的CNN情感分类方法有较大提升。

实验环境：
---

LINUX
Python3.6 以上
Tensorflow1.2 以上
Cuda8

运行：
----

首先修改路径：
文件路径配置均在text_model.py文件中、data中为原始数据、cleanning_data中是经过处理的数据
LDA_model_path在LDA_train、LDA_vec中
save_dir、tensorboard_dir在text_train、text_test
测试：
Python yourpath/text_cnn_LDA_word2vec/text_test.py
训练分类器：
Python  yourpath/text_cnn_LDA_word2vec/text_train.py
训练LDA模型（LDA_pretrain中的是经LAD模型预训练的数据）：
Python  yourpath/text_cnn_LDA_word2vec/LDA_train.py
训练Word2vec模型（word2vec_pretrain中的是经word2vec模型预训练的数据）：
Python  yourpath/text_cnn_LDA_word2vec/ train_word2vec.py
