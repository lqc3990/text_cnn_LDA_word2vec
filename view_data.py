import pandas as pd
import random
df_train = pd.read_csv('drive/data/sentiment_analysis_trainingset.csv',index_col='id')
df_val = pd.read_csv('drive/data/sentiment_analysis_validationset.csv',index_col='id')


print('train训练集的值分布：\n',df_train['others_overall_experience'].value_counts())
print('val验证集值分布：\n',df_val['others_overall_experience'].value_counts())

#合并两个dataframe
df=[df_train,df_val]
#index 列自动重置
df=pd.concat(df,ignore_index= True)

print('调整前的数据集值分布：\n',df['others_overall_experience'].value_counts())

df_1=df[df['others_overall_experience']==1]
df_0=df[df['others_overall_experience']==0]
df_f1=df[df['others_overall_experience']==-1]
df_f2=df[df['others_overall_experience']==-2]

df_0=df_0.sample(n=10000)
df_1=df_1.sample(n=10000)
df_f1=df_f1.sample(n=10000)

df=[df_1,df_0,df_f1]
df=pd.concat(df,ignore_index= True)
print('调整后的数据集值分布：\n',df['others_overall_experience'].value_counts())

#随机打乱顺序
df_0=df_0.sample(frac=1)
df_f1=df_f1.sample(frac=1)
df_1=df_1.sample(frac=1)

content_0=list(df_0['content'])
label_0=list(df_0['others_overall_experience'])

content_1=list(df_1['content'])
label_1=list(df_1['others_overall_experience'])

content_f1=list(df_f1['content'])
label_f1=list(df_f1['others_overall_experience'])

for i in range(len(content_0)):
    content_0[i]=str(content_0[i].replace('\n','').replace(' ','').replace('"','').replace('\t',''))
    content_0[i]='__label__'+str(label_0[i])+'\t'+content_0[i]+'\n'

for i in range(len(content_1)):
    content_1[i]=str(content_1[i].replace('\n','').replace(' ','').replace('"','').replace('\t',''))
    content_1[i]='__label__'+str(label_1[i])+'\t'+content_1[i]+'\n'
	
for i in range(len(content_f1)):
    content_f1[i]=str(content_f1[i].replace('\n','').replace(' ','').replace('"','').replace('\t',''))
    content_f1[i]='__label__'+str(label_f1[i])+'\t'+content_f1[i]+'\n'	

content_train=[]
content_val=[]
content_test=[]

content_train.extend(content_0[0:6999])
content_train.extend(content_1[0:6999])
content_train.extend(content_f1[0:6999])

content_val.extend(content_0[7000:7999])
content_val.extend(content_1[7000:7999])
content_val.extend(content_f1[7000:7999])

content_test.extend(content_0[8000:9999])
content_test.extend(content_1[8000:9999])
content_test.extend(content_f1[8000:9999])
random.shuffle (content_train)
random.shuffle (content_val)
random.shuffle (content_test)

#val_proportion=0.14


f = open('drive/kechengdata/val.txt','w',encoding='utf-8')
#f.writelines(content[0:int(len(content)*val_proportion)])
f.writelines(content_val)
f.close()

f = open('drive/kechengdata/train.txt','w',encoding='utf-8')
#f.writelines(content[int(len(content)*val_proportion):])
f.writelines(content_train)
f.close()

f = open('drive/kechengdata/test.txt','w',encoding='utf-8')
#f.writelines(content[int(len(content)*val_proportion):])
f.writelines(content_test)
f.close()









