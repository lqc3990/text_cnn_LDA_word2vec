#-*- coding : utf-8 -*-
# coding: utf-8
import sys
import os
import jieba
#from mpltools import style
import numpy as np
#from mpltools import style
from collections import  Counter
import tensorflow.contrib.keras as kr
import matplotlib.pyplot as plt
import importlib
import re
import codecs
from text_model import TextConfig
importlib.reload(sys)
# 设置中文环境#
#reload(sys)
#sys.setdefaultencoding("utf-8")
#去除体用词
def get_stop_words(file_name):
    with open(file_name,'r') as file:
        return set([line.strip() for line in file])

def read_file(filename,outname,stpwrdpath):
    """
    Args:
        filename:trian_filename,test_filename,val_filename 
    Returns:
        two list where the first is lables and the second is contents cut by jieba
        
    """
    #stopwords=config.stpwrdpath
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents,labels=[],[]
    stpwrd_dic = codecs.open(stpwrdpath,'r',encoding='utf-8')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    num=0
    
    with codecs.open(filename,'r',encoding='utf-8') as f:
        outputs = open(outname, 'w',encoding='utf-8')
        for line in f:
              #print(num)
                
                #line=line.rstrip()
              try:
                #label,content=line.split('\t')
                line=line.split('\t')
                assert len(line)==2
                label=line[0]
                content=line[1]
                label=label.encode('utf-8').decode('utf-8-sig')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                
                outstr = ''
                outstr +=label
                outstr +='\t'
                for blk in blocks:
                    if re_han.match(blk):
                        new_line=jieba.cut(blk)
                        
                        for word in new_line:
                            if word not in stpwrdlst:
                               if word != '\t':
                                  outstr += word
                                  outstr += " "
                outstr += "\n"     
                num=num+1
                #print(num)
                outputs.write(outstr)
              except:
                pass
        outputs.close() 

    #return labels,contents
if __name__ == '__main__':
   config=TextConfig()
   read_file(config.val_filename,config.val_w2v_data,config.stpwrdpath)
   read_file(config.train_filename,config.train_w2v_data,config.stpwrdpath)
   read_file(config.test_filename,config.test_w2v_data,config.stpwrdpath)
   '''
   f = open('drive/data/fenci_data.txt','w',encoding='utf-8')
   for line in  contents:
       
       f.write(str(line))
           
       f.write("\n")
   f.close()
   doclist=[]
  '''
   '''
   f = open('drive/data/val_fenci.txt','w',encoding='utf-8')
   f.writelines(content[0:int(len(content)*0.14)])
   f.close()
   '''

