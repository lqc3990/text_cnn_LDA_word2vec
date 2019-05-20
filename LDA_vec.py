#-*- coding : utf-8 -*-
# coding: utf-8
import sys
import importlib
importlib.reload(sys)
import os
import gc
import codecs
from text_model import  *
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

if __name__ == '__main__':
    config=TextConfig()
    doclist=[]
    k=[]
    num=0;
    middatafolder ='drive/text_cnn_LDA_word2vec/lda_model/'#change drive/  to  yourpath
    dictionary_path = middatafolder + 'dictionary.dictionary' 

    with codecs.open(config.train_LDA_data, 'r', code) as source_file:
         for line in source_file: 
             doclist.append(line.split(' '))
    lda = models.ldamodel.LdaModel.load(middatafolder +'lda_q.model') 
    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = [dictionary.doc2bow(content) for content in doclist]
    #key:get probility of topic for every ducument
    a=lda.get_document_topics(corpus, minimum_probability=0, minimum_phi_value=0, per_word_topics=False)
    c=[[x[1] for x in b] for b in a]
    numpy_array = np.array(c)
    np.save(config.pre_trianing_LDAtrain,numpy_array )
    numpy_arrays = np.load(config.pre_trianing_LDAtrain)
    a=numpy_arrays.tolist()
    #np.save('drive/data/LDA_vec.npy',numpy_array[0:int(len(a)*0.86] )
    print (a[0])
    print (a[1])
    print (a[2])

    

    with codecs.open(config.val_LDA_data, 'r', code) as source_file:
         for line in source_file: 
             doclist.append(line.split(' '))
    lda = models.ldamodel.LdaModel.load(middatafolder +'lda_q.model') 
    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = [dictionary.doc2bow(content) for content in doclist]
    a=lda.get_document_topics(corpus, minimum_probability=0, minimum_phi_value=0, per_word_topics=False)
    c=[[x[1] for x in b] for b in a]
    numpy_array = np.array(c)
    np.save(config.pre_trianing_LDAval,numpy_array )

    with codecs.open(config.test_LDA_data, 'r', code) as source_file:
         for line in source_file: 
             doclist.append(line.split(' '))
    lda = models.ldamodel.LdaModel.load(middatafolder +'lda_q.model') 
    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = [dictionary.doc2bow(content) for content in doclist]
    a=lda.get_document_topics(corpus, minimum_probability=0, minimum_phi_value=0, per_word_topics=False)
    c=[[x[1] for x in b] for b in a]
    numpy_array = np.array(c)
    np.save(config.pre_trianing_LDAtest,numpy_array )

    
    ''''
    # read probility of topic for every ducument file
    
    with codecs.open(middatafolder + 'topic.txt', 'r', code) as source_file:
         for line in source_file:
             k.append(line.split(' '))
         print(k)
    '''
    #画主题-单词分布
    '''
    num_show_term = 50   # 每个主题下显示几个词
    for topic_id in range(num_topics):
        print('第%d个主题的词与概率如下：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        #print('词：\t', end='  ')
        #for t in term_id:
            #print(dictionary.id2token[t], end=' ')
        print("{}".format(lda.print_topic(topic_id)))
        #print('\n概率：\t', term_distribute[:, 1])
     '''