# -*- coding: utf-8 -*-document
"""
Created on Tue Jun 14 00:18:17 2022

@author: Qingwen Liang
"""

"""
说明：
本程序用于计算文档参数 InnoDis,InnoTone,Cwords,InnoSimi

文档经jieba分词为单词序列
创新关键词汇表（创新关键词.xlsx），由word2vec_model+种子词汇生成
停用词（stop_words.txt),来自网络整理资源
"""

   
import jieba
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import re
from keras.preprocessing.text import Tokenizer

file_path = './07-20年报/'
file_names = os.listdir(file_path)
file_names = [file_path+file_name for file_name in file_names]
file_count = len(file_names)#总文档数
    
dictname = '金融领域中文情绪词典.xlsx'
pos_dict = pd.read_excel(dictname,sheet_name='年报正面')['pos'].tolist()
neg_dict = pd.read_excel(dictname,sheet_name='年报负面')['neg'].tolist()

# key_words = ['核心竞争力','创新','研发','专利']
key_words = pd.read_excel("创新关键词.xlsx")['keywords'].tolist()
key_words = [eval(keyword[:-1])[0] for keyword in key_words]
for word in key_words:
    jieba.add_word(word)

#停用词列表
stop_words_path = './stop_words.txt'
stop_words = [w.strip() for w in open(stop_words_path)]


#-------------------------------------------------------------------------------------------
"""
用于文档向量化的 Tokenizer
"""
#文档流生成器
def file_generator(file_list):
    i = 0#打印进度
    for file in file_list:
        try:
            with open(file,'r',encoding="UTF-16") as open_file:
                text = open_file.read()
                #正则去除字母、数字及特殊字符，只保留中文汉字
                pat = re.compile('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s：；（）+±•÷○]+')                 
                text = pat.sub("",text)
                words = jieba.lcut_for_search(text)
                words = [w for w in words if w not in stop_words]    
                yield words
            
        except(Exception):
            print("Document parsing error:",file)
        i += 1
        print("\rtokenizer fit on texts : {:.2%} completed ".format(i/file_count),end='',flush=True)
          
file_gen = file_generator(file_names)
# for w in file_gen:
#     print(w[0])

vocabulary_size = 20000#文档向量化仅考察出现频率前vocabulary_size的词汇
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(file_gen)
word_counts = tokenizer.word_counts
vocab_length = len(word_counts)#总词汇数
# words_list = list(word_counts)#总词汇列表
# word_index = tokenizer.word_index
# word_counts_list = list(word_counts)
# word_docs = tokenizer.word_docs
print('\n')  

#-------------------------------------------------------------------------------------
"""
计算文档参数 InnoDis,InnoTone,Cwords,Vector
"""
InnoDis = [0]*file_count
InnoTone = [0]*file_count
Cword = [0]*file_count
vector = np.zeros((file_count,vocabulary_size))

for i in range(file_count):
    file = file_names[i]
    # print(file)
    try:
        with open(file,'r',encoding="UTF-16") as open_file:
          text = open_file.read()
          
          """
          文档预处理数据
          """
          #正则去除字母、数字及特殊字符，只保留中文汉字
          pat = re.compile('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s：；（）+±•÷○]+')                 
          text = pat.sub("",text)
          words = jieba.lcut_for_search(text)
          words = [w for w in words if w not in stop_words]
          len_words = len(words)

          vocab_i = defaultdict(int)#第i篇文档词频字典
          mask = np.zeros(len_words)
          key_count = 0
          for j in range(len_words):
              word = words[j]
              if word in word_counts:
                  vocab_i[word] += 1
              if word in key_words:
                  key_count += 1 #关键词频+1
                  mask[j-100:j+100] = 1 #标记关键词前后100个词
                  # print(words[j])
          """
          InnoDis计算:
              由关键词汇数/文档总词数计算
          """
          InnoDis[i] = key_count/len_words*1000
          # print("InnoDis = ",key_count,'/',len_words)
          
          """
          InnoTone计算:
              由关键词定位文档相关内容（前后100词）作为考察文本，统计考察文本内的
          积极词和消极词（“金融领域中文情绪词典.xlsx”）个数，进而计算情感分数              
          """
          words_marked = []  #需考察的词汇列表
          for j in range(len_words):
              if mask[j]:
                  words_marked.append(words[j])
          pos_count = 0 #积极词汇数
          neg_count = 0 #消极词汇数
          for word_marked in words_marked:
              if word_marked in pos_dict:
                  pos_count += 1
              if word_marked in neg_dict:
                  neg_count += 1
          if (pos_count+neg_count) == 0:
              InnoTone[i] = np.nan
          else:
              InnoTone[i] = (pos_count-neg_count)/(pos_count+neg_count)
          
          """
          Cword计算：
              由本文档词频字典vocab_i与全体文档词频字典vocab_tf对应项相乘，并除以全文档词数计算
          """
          for word in vocab_i:
              Cword[i] += vocab_i[word]*word_counts[word]
          Cword[i] /= sum([v for v in vocab_i.values()])
          Cword[i] = np.log(Cword[i])
          
          """
          Vector计算：
              由关键词定位文档相关内容（前后100词）作为考察文本，即上面计算出的words_marked，
          利用tokenizer计算TF-IDF向量(IDF加权的词袋向量)作为文档的特征向量Vector
          """
          vector[i] = tokenizer.texts_to_matrix([' '.join(words_marked)],mode='tfidf').reshape(-1)
          
    except(Exception) as e:
        print("\nDocument parsing error:", e, file)
        InnoDis[i] = np.nan
        InnoTone[i] = np.nan
        Cword[i] = np.nan
        vector[i] = np.nan

    print("\rIndexes computing : {:.2%} completed ".format((i+1)/file_count),end='',flush=True)
#-------------------------------------------------------------------------------------------
"""
生成参数表
"""
stock_pat = re.compile(r"_(\d{6})")
year_pat = re.compile(r"(\d{4})年")
stock = [stock_pat.search(file_name).group(1) if stock_pat.search(file_name) else np.nan for file_name in file_names]
year = [year_pat.search(file_name).group(1) if stock_pat.search(file_name) else np.nan for file_name in file_names]

lst_data = [row for row in zip(stock,year,InnoDis,InnoTone,Cword,vector)]
df_data = pd.DataFrame(lst_data,columns=['stock','year','InnoDis','InnoTone','Cwords','vector'])

df_data.dropna(how='any',inplace=True) #去除含有nan的数据行
df_data.sort_values(by=['stock','year'],inplace=True)
df_data.drop_duplicates(subset=['stock','year'],keep='last',inplace=True)#去除同一只股票同年的重复数据
df_data.reset_index(drop=True,inplace=True)

"""
由文档特征vector，计算当年和上一年的年报相似度InnoSimi
"""
vector_array = df_data['vector'].to_list()
vector_norm = np.asarray([np.linalg.norm(v) for v in vector_array])
vector_normalized = vector_array/vector_norm[:,np.newaxis]
# vector_normalized_norm = np.asarray([np.linalg.norm(v) for v in vector_normalized])
simi = [np.dot(a,b) for a,b in zip(vector_normalized[:-1],vector_normalized[1:])]
simi[0:0] = [np.nan]

df_data['InnoSimi'] = simi
df_data['InnoSimi'].where(df_data['stock'] == df_data['stock'].shift(1), np.nan,inplace=True)

#-----------------------------------------------------------------------------------------
"""
输出参数到excel文件
"""

df_data[['stock','year','InnoDis','InnoTone','Cwords','InnoSimi']].to_excel('./Indexes.xlsx')
print("Data has been written to Indexes.xlsx")


