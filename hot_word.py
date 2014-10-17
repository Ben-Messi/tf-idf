#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import datetime
import random
import pickle
from bitarray import bitarray
import re
import os
import jieba
import math
import ujson
import cPickle
import heapq
import copy
import leveldb
import fast_search

'''
    独立出idf类, 如果某个词不在idf词典和停用词词典中 应给予高idf
    每篇文章分出词的前n%作为文章特征词, 依次记录到词dict中
    每个分类一个对象, 指定不同的leveldb
    首先实现热词记录类
    '........'字符串分词结果会以其本身作为一个词
    以停用词作为词典, 对idf类中没有的词做fast_search
    如果找到 则作为垃圾词,否则调高idf值

    如果找到如: t.cn/url.cn/dwz.cn类的短网址domain
    首先使用正则替换掉"http://t.cn/XXXXXX"
    防止XXXXX切出词, 被当做高idf值的词, 占用词典空间
    因为微博数据很多，不加此过滤会导致词典大小暴增
    \b(http:\/\/)?[\w\d]+\.[\w\d\.]+\/[\w\d_!@#$%^&\*-_=\+]+
'''

class idf:
    def __init__(self, dics = {}):
        self.word_dic = dics
        self.fcounter = 0
        self.default_idf = 10
        self.log_base = math.e
        self.rubbish_set, self.rubbish_hd = self.get_rubbish_set()
        jieba.initialize()

    def get_rubbish_set(self, stopword_path = "stopwords.txt"):
        rubbish_set = set()
        hd = 0
        try:
            hd = fast_search.load(stopword_path)
            with open(stopword_path, "r") as fd:
                for l in fd:
                    rubbish_set.add(l.strip())
        except:
            hd = 0
            #如果读不到文件则忽略
            pass
        rubbish_set.add(" ")
        rubbish_set.add("　")
        rubbish_set.add("\t")
        rubbish_set.add("\r")
        rubbish_set.add("\n")
        rubbish_set.add("\r\n")
        rubbish_set.add("DC")
        rubbish_set.add("DS")
        rubbish_set.add("gt")
        return rubbish_set, hd

    def set_rubbish_set(self, rubbish_set):
        self.rubbish_set = rubbish_set

    def is_rubbish(self, word):
        if word in self.rubbish_set:
            return 1
        if self.word_dic.has_key(word):
            return 0
        result = fast_search.findall(self.rubbish_hd, word)
        if result:
            return 1
        else:
            return 0

    def set_log_base(self, log_base):
        """
        计算idf值log的底默认为math.e
        此函数用于更改默认的底
        """
        self.log_base = log_base

    def set_default_idf(self, idf):
        """
        对不存在的词idf值默认返回10
        此函数用于更改默认的idf返回值
        """
        self.default_idf = idf

    def add_doc(self, s):
        word_set = set(jieba.cut(s))
        for w in word_set:
            word = w.encode("utf-8")
            if not self.word_dic.has_key(word):
                self.word_dic[word] = 0
            self.word_dic[word] += 1
        self.fcounter += 1
    
    def get_idf(self, word):
        """
        如果此word不存在 则返回default
        使用set_default_idf(idf) 设置默认idf值
        """
        if word in self.rubbish_set:
            return math.log(float(self.fcounter)/self.word_dic[word], self.log_base)
        elif word not in self.word_dic:
            return self.default_idf
        else:
            return math.log(float(self.fcounter)/self.word_dic[word], self.log_base)

    def loads(self, s):
        dics = pickle.loads(s)
        self.set_vars_by_dics(dics)

    def dumps(self):
        dics = self.gen_dics()
        s = cPickle.dumps(dics)
        return s
    
    def save_idf_by_fpath(self, fpath):
        with open(fpath, "w") as fd:
            self.save_idf_by_fd(fd)

    def save_idf_by_fd(self, fd):
        idf_list = []
        for w in self.word_dic:
            lidf = self.get_idf(w)
            idf_list.append((lidf, w))
        idf_list = heapq.nlargest(len(idf_list), idf_list)
        for i in idf_list:
            fd.write(i[1] + "\t" + str(i[0]) + "\n")

    def set_vars_by_dics(self, dics):
        self.word_dic    = dics["word_dic"]
        self.fcounter    = dics["fcounter"]
        self.default_idf = dics["default_idf"]
        self.log_base    = dics["log_base"]
        self.rubbish_set = dics["rubbish_set"]
    
    def gen_dics(self):
        dics = {}
        dics["word_dic"]    = self.word_dic
        dics["fcounter"]    = self.fcounter
        dics["default_idf"] = self.default_idf
        dics["log_base"]    = self.log_base
        dics["rubbish_set"] = self.rubbish_set
        return dics

class hot_word:
    def __init__(self, db_name):
        self.idf_hd = idf()
        with open("idf_dumps.txt", "r") as fd:
            s = fd.read()
    
        self.idf_hd.loads(s)
        self.hot_word_dic = {}
        self.short_url_hd = fast_search.load("short_url.txt")
        self.dbhd = leveldb.LevelDB(db_name)
        self.url_re = re.compile(r'(http:\/\/)*[\w\d]+\.[\w\d\.]+\/[\w\d_!@#$%^&\*-_=\+]+')
        #内部使用batch做缓存 add_doc时暂时不写入db文件
        #要获取结果，或者达到阈值(batch_limit)时才写入文件
        self.batch = leveldb.WriteBatch()
        self.batch_counter = 0
        self.batch_limit = 100000
        self.fid = 0
        #self.get_file_word_flag = "percent"
        self.get_file_word_flag = "num"
        self.word_list_n = 5
        self.get_file_word_cbk = {}
        self.get_file_word_cbk["num"] = self.get_file_word_list_by_num
        self.get_file_word_cbk["percent"] = self.get_file_word_list_by_persent

    def set_options(self, option_dic):
        '''
        get_word_flag:('num', 'percent') 表示使用哪种方式取文章特征词
            'num'取tf-idf排名前 n的词, 
            'percent' 取tf-idf排名前 n%的词
        'word_top_num':[1, int_max) 如果使用方式 'num' n的值
        'word_top_persent':[1, 100] 如果使用方式 'percent' n的值
        '''
        self.get_file_word_flag = option_dic.get("get_word_flag", "num")
        self.word_list_num = option_dic.get("word_top_num", 5)
        self.word_list_persent = option_dic.get("word_top_persent", 10)
        self.batch_limit = option_dic.get("batch_limit", 100000)
        if self.get_file_word_flag == "percent":
            self.word_list_n = self.word_list_persent
        elif self.get_file_word_flag == "num":
            self.word_list_n = self.word_list_num
        else:
            self.word_list_n = 0

    def dumps(self):
        dics = {}
        dics["idf_hd"] = self.idf_hd.dumps()
        dics["hot_word_dic"] = pickle.dumps(self.hot_word_dic)
        self.write_and_clean_batch()
        s = pickle.dumps(dics)
        return s
    
    def loads(self, s):
        dics = pickle.loads(s)
        self.idf_hd = idf()
        self.idf_hd.loads(dics["idf_hd"])
        self.hot_word_dic = pickle.loads(dics["hot_word_dic"])

    def get_file_word_list_by_num(self, s, n):
        '''
        获取文章tf-idf值top_n的词列表
        '''
        word_list = list(jieba.cut(s))

        return self.get_file_word_list_base(word_list, n)

    def get_file_word_list_by_persent(self, s, n):
        '''
        获取文章tf-idf值top n%的词列表
        '''
        word_list = list(jieba.cut(s))
        #使用去重后的词数
        word_num = int(len(set(word_list)) * n * 0.01)
        if not word_num:
            word_num = 1
        #print "%d:%d " % (word_num, len(word_list))

        return self.get_file_word_list_base(word_list, word_num)

    def get_file_word_list_base(self, word_list, n):
        '''
        传入分好的文章词列表
        '''
        l_word_dic = {}
        for w in word_list:
            word = w.encode("utf-8")
            if self.idf_hd.is_rubbish(word):
                continue
            if not l_word_dic.has_key(word):
                l_word_dic[word] = 0
            l_word_dic[word] += 1
        ret_list = []
        for word in l_word_dic:
            tf_idf = l_word_dic[word] * self.idf_hd.get_idf(word)
            ret_list.append((tf_idf, word))
        l = heapq.nlargest(n, ret_list)
        return l

    def s_filter(self, s):
        result = fast_search.findall(self.short_url_hd, s)
        if result:
            s = self.url_re.sub("", s)
        return s

    #此doc不再加到idf中, idf不再变化
    def add_doc(self, fname):
        try:
            with open(fname, "r") as fd:
                s = fd.read()
        except:
            return -1
        
        self.add_doc_s(s)
    
    def add_doc_s(self, s):
        self.batch.Put(str(self.fid), s)
        self.batch_counter += 1
        if self.batch_counter > self.batch_limit:
            self.write_and_clean_batch()
        s = self.s_filter(s)
        word_list = self.get_file_word_cbk[self.get_file_word_flag](s, self.word_list_n)
        for w in word_list:
            word = w[1]
            if not self.hot_word_dic.has_key(word):
                self.hot_word_dic[word] = set()
            self.hot_word_dic[word].add(self.fid)
        self.fid += 1

    def get_top_n_word_list(self, n):
        hot_word_list = []
        for word in self.hot_word_dic:
            #print self.hot_word_dic[word], word
            hot_word_list.append((len(self.hot_word_dic[word]), word))
        l = heapq.nlargest(n, hot_word_list)
        return l
    
    def write_and_clean_batch(self):
        if self.batch_counter:
            self.dbhd.Write(self.batch)
            self.batch_counter = 0
            self.batch = leveldb.WriteBatch()

    def get_file_by_fid(self, fid):
        self.write_and_clean_batch()
        try:
            s = self.dbhd.Get(str(fid))
        except:
            s = ""
        return s
    
    def get_file_list_by_word(self, word):
        fid_list = self.hot_word_dic.get(word, [])
        for fid in fid_list:
            yield self.get_file_by_fid(fid)
