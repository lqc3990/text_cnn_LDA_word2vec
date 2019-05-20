#-*- coding : utf-8 -*-
# coding: utf-8
import sys
import importlib
importlib.reload(sys)
import os
import gc
import codecs
from text_model import *
from gensim.corpora import Dictionary
from gensim import corpora, models
from datetime import datetime

import platform
import logging
from loader import *
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s : ', level=logging.INFO)

platform_info = platform.platform().lower()
if 'windows' in platform_info:
    code = 'gbk'
elif 'linux' in platform_info:
    code = 'utf-8'
path = sys.path[0]

class GLDA(object):
    """docstring for GdeltGLDA"""

    def __init__(self, stopfile=None):
        super(GLDA, self).__init__()
        if stopfile:
            with codecs.open(stopfile, 'r', code) as f:
                self.stopword_list = f.read().split(' ')
            print ('the num of stopwords is : %s'%len(self.stopword_list))
        else:
            self.stopword_list = None

    def lda_train(self, num_topics, datafolder, middatafolder, dictionary_path=None, corpus_path=None, iterations=None, passes=None, workers=3):       
        #labels,contents=read_file('drive/data/test.txt')
        time1 = datetime.now()
        num_docs = 0
        doclist = []
        
        if not corpus_path or not dictionary_path: # 若无字典或无corpus，则读取预处理后的docword。一般第一次运行都需要读取，在后期调参时，可直接传入字典与corpus路径
            #for filename in os.listdir(datafolder): # 读取datafolder下的语料for filename in self.filenames:
            for filename in datafolder:
                with codecs.open(filename, 'r', code) as source_file:
                    for line in source_file: 
                    
                        num_docs += 1
                        if num_docs%100000==0:
                            print ('%s, %s'%(datafolder, num_docs))
                        #doc = [word for word in doc if word not in self.stopword_list]
                        doclist.append(line.split(' '))
                        #print (line)
                    del line
                    gc.collect()
                        #print (line)
                    print ('%s, %s'%(datafolder, num_docs))
                  
        if dictionary_path:
            dictionary = corpora.Dictionary.load(dictionary_path) # 加载字典
        else:            
            #构建词汇统计向量并保存
            dictionary = corpora.Dictionary(doclist)
            dictionary.save(middatafolder + 'dictionary.dictionary')
        if corpus_path:
            corpus = corpora.MmCorpus(corpus_path) # 加载corpus
        else:
            corpus = [dictionary.doc2bow(doc) for doc in doclist]
            corpora.MmCorpus.serialize(middatafolder + 'corpus.mm', corpus) # 保存corpus
        tfidf = models.TfidfModel(corpus)
        corpusTfidf = tfidf[corpus]
        #del corpus
        #gc.collect()
        time2 = datetime.now()
        lda_multi = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, \
            iterations=iterations, workers=workers, batch=True, passes=passes) # 开始训练
        del dictionary
        gc.collect()
        lda_multi.print_topics(num_topics, 20) # 输出主题词矩阵
        print ('lda training time cost is : %s, all time cost is : %s '%(datetime.now()-time2, datetime.now()-time1))
        #模型的保存/ 加载
        #lda_multi.save(middatafolder + 'lda_tfidf_%s_%s.model'%(2014, num_topics, iterations)) # 保存模型
        lda_multi.save(middatafolder + 'lda_q.model')
        # lda = models.ldamodel.LdaModel.load('zhwiki_lda.model') # 加载模型
        # save the doc-topic-id
        topic_id_file = codecs.open(middatafolder + 'topic.txt', 'w', 'utf-8')
        '''
        for i in range(num_docs):
            #topic_id = lda_multi[corpusTfidf[i]][0][0] # 取概率最大的主题作为文本所属主题
            topic_id = lda_multi[corpusTfidf[i]]
            topic_id_file.write(str(topic_id)+ '\n')
        '''   
if __name__ == '__main__':
    config = TextConfig()
    train_LDA_data=config.train_LDA_data  #train data
    test_LDA_data=config.test_LDA_data    #test data
    val_LDA_data=config.val_LDA_data     #validation data
    datafolder=[train_LDA_data,test_LDA_data,val_LDA_data]

    middatafolder ='drive/text_cnn_LDA_word2vec/lda_model/'#change drive/  to  yourpath

    dictionary_path = middatafolder + 'dictionary.dictionary' # 已处理好的字典，若无，则设置为False
    corpus_path = middatafolder + 'corpus.mm' # 对语料处理过后的corpus，若无，则设置为False

    num_topics =500
    passes = 100 # 这个参数大概是将全部语料进行训练的次数，数值越大，参数更新越多，耗时更长
    iterations = 6000
    workers = 3 # 相当于进程数
    
    lda = GLDA()
    lda.lda_train(num_topics, datafolder, middatafolder, dictionary_path=False, corpus_path=False, iterations=iterations, passes=passes, workers=workers)
 