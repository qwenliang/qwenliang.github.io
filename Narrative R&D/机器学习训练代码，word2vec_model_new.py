# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:34:07 2022

@author: Qingwen Liang
"""
import os
import jieba
import gensim
import re
import pprint



#停用词列表
stop_words_path = './stop_words.txt'
stop_words = [w.strip() for w in open(stop_words_path)]

#语料库类
class MyCorpus:
    # an iterator that yield sentence(list of str)
    def __init__(self,file_path):
        self.file_path = file_path
        
    def __iter__(self):
        file_names = os.listdir(self.file_path)
        file_names = [self.file_path+file_name for file_name in file_names]
        for file in file_names:
            print("training models: \nprocessing ",file)
            try:
                with open(file,'r',encoding="UTF-8") as f:
                    doc = f.read()
                    doc = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s：；（）]+', "", doc)
                    doc_list = jieba.lcut_for_search(doc)
                    doc_list = [w for w in doc_list if w not in stop_words]
                    yield doc_list
            except(Exception):
                print("Document parsing error")
                continue

#准备语料库
file_path1 = './document/'            
sentences1 = MyCorpus(file_path1)

#训练模型
model_path = './model/my_model'
if os.path.exists(model_path):
    model = gensim.models.Word2Vec.load(model_path)
else:
    model = gensim.models.Word2Vec(sentences = sentences1)
    os.mkdir('./model')
    model.save('./model/my_model')

#继续训练模型 document_new 是语料库所在位置
file_path2 = './document_new/'    
sentences2 = MyCorpus(file_path2)
model.build_vocab(sentences2,update = True)
model.train(sentences2,total_examples=model.corpus_count,epochs=model.epochs)
model.save('./model/my_model')
    
#获取词向量结果
wv = model.wv
vectors = wv.vectors
# word_list = wv.index2word
word_list = wv.index_to_key
#种子词汇
seed_word = ['技术创新','研究','开发','研发','专利','发明']
seed_word = [w for w in seed_word if w in word_list]

#获取相似词，前topn个
similary_words = model.wv.most_similar(seed_word, topn=700)

#import keras 














