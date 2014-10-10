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

def gen_rubbish_set():
    rubbish_set = set()
    with open("/home/kelly/code/warning/key/stopwords.txt", "r") as fd:
        for l in fd:
            rubbish_set.add(l.strip())
    rubbish_set.add(" ")
    rubbish_set.add("　")
    rubbish_set.add("\t")
    rubbish_set.add("\r")
    rubbish_set.add("\n")
    rubbish_set.add("\r\n")
    rubbish_set.add("DC")
    rubbish_set.add("DS")
    rubbish_set.add("gt")
    return rubbish_set

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

class top_n:
    def __init__(self, n):
        self.n = n
        self.hp = []

    def add_item(self, item):
        if len(self.hp) >= self.n:
            heapq.heappushpop(self.hp, item)
        else:
            heapq.heappush(self.hp, item)

    def get_sortted_list(self):
        return heapq.nlargest(self.n, self.hp)
    
    def clear(self):
        self.hp = []
     
    def resize(self, n):
        if n > self.n:
            self.n = n
        else:
            while len(self.hp) > n:
                heapq.heappop(self.hp)

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
            return 0
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

class tf_idf:
    def __init__(self, idf_path, stop_words_path = ""):
        self.idf_dic = self.gen_idf_dic(idf_path)
        self.rubbish_set = self.gen_rubbish_set(stop_words_path)
    
    def gen_idf_dic(self, idf_path):
        ret_dic = {}
        with open(idf_path) as fd:
            for l in fd:
                idx = l.find("\t")
                word = l[:idx]
                try:
                    l_idf = float(l[idx + 1:].strip())
                except:
                    continue
                ret_dic[word] = l_idf
        return ret_dic
    
    def set_rubbish_set(self, rubbish_set):
        self.rubbish_set = rubbish_set

    def gen_rubbish_set(self, stop_words_path):
        ret_set = set()
        if not stop_words_path:
            return ret_set
        with open(stop_words_path) as fd:
            for l in fd:
                ret_set.add(l.strip())
        return ret_set
    
    def get_top_n_tf_idf(self, doc):
        word_list = list(jieba.cut(doc))
        top_n_list = self.get_top_n_word_list(word_list)
        return [i[1] for i in top_n_list]
    
    def get_top_n_word_list(self, word_list):
        '''
        传入分好的文章词列表
        '''
        l_word_dic = {}
        for w in word_list:
            word = w.encode("utf-8")
            if word in self.rubbish_set:
                continue
            if not l_word_dic.has_key(word):
                l_word_dic[word] = 0
            l_word_dic[word] += 1
        ret_list = []
        for word in l_word_dic:
            tf_idf = l_word_dic[word] * self.idf_dic.get(word, 0)
            ret_list.append((tf_idf, word))
        l = heapq.nlargest(5, ret_list)
        return l

class tf_idf_old:
    def __init__(self, dics = None):
        '''
        fname_dic = {fid:set([word list])}
        word_dic = {'word':{fid:tf, ...}}
        '''
        if dics:
            self.set_vars_by_dics(dics)
        else:
            self.word_dic = {}
            self.fname_dic = {}
            self.fcounter = 0
            self.default_idf = 0
            self.log_base = math.e
            self.rubbish_set = set()
            self.proportion = 0.3
        jieba.initialize()

    def set_rubbish_set(self, rubbish_set):
        self.rubbish_set = rubbish_set

    def set_log_base(self, log_base):
        """
        计算idf值log的底默认为math.e
        此函数用于更改默认的底
        """
        self.log_base = log_base

    def set_default_idf(self, idf):
        """
        对不存在的词idf值默认返回0
        此函数用于更改默认的idf返回值
        """
        self.default_idf = idf

    def add_doc(self, fid, s):
        word_list = list(jieba.cut(s))
        for w in word_list:
            word = w.encode("utf-8")
            if not self.word_dic.has_key(word):
                self.word_dic[word] = {}
            if not self.word_dic[word].has_key(fid):
                self.word_dic[word][fid] = 1
            else:
                self.word_dic[word][fid] += 1
        self.fname_dic[fid] = set(word_list)
        self.fcounter += 1

    def get_idf(self, word):
        """
        如果此word不存在 则返回default
        使用set_default_idf(idf) 设置默认idf值
        """
        if word not in self.word_dic:
            return self.default_idf
        else:
            return math.log(float(self.fcounter)/len(self.word_dic[word]), self.log_base)

    def get_tf_idf_by_file(self, word, fid):
        idf = self.get_idf(word)
        tf = self.word_dic[word][fid]
        return tf * idf

    def get_tf_idf_list_by_str(self, s):
        word_list = list(jieba.cut(s))
        l_word_dic = {}
        for w in word_list:
            word = w.encode("utf-8")
            if word in self.rubbish_set:
                continue
            if not l_word_dic.has_key(word):
                l_word_dic[word] = 0
            l_word_dic[word] += 1
        ret_list = []
        for word in l_word_dic:
            tf_idf = l_word_dic[word] * self.get_idf(word)
            ret_list.append((tf_idf, word))
        return heapq.nlargest(len(ret_list), ret_list)

    def set_vars_by_dics(self, dics):
        self.word_dic = dics["word_dic"]
        self.fname_dic = dics["fname_dic"]
        self.fcounter = dics["fcounter"]
        self.default_idf = dics["default_idf"]
        self.log_base = dics["log_base"]
        self.rubbish_set = dics["rubbish_set"]
        self.proportion = dics.get("proportion", 0.3)

    def gen_dics(self):
        dics                = {}
        dics["word_dic"]    = self.word_dic
        dics["fname_dic"]   = self.fname_dic
        dics["fcounter"]    = self.fcounter
        dics["default_idf"] = self.default_idf
        dics["log_base"]    = self.log_base
        dics["rubbish_set"] = self.rubbish_set
        dics["proportion"]  = self.proportion
        return dics

    def dumps(self):
        dics = self.gen_dics()
        s = cPickle.dumps(dics)
        return s

    def loads(self, s):
        dics = cPickle.loads(s)
        self.set_vars_by_dics(dics)
        return 0

    def get_max_tf_idf(self):
        max_tf_idf = (0, 0, 0, "")
        word_dic = self.word_dic
        for w in word_dic:
            if w in self.rubbish_set:
                continue
            for fid in word_dic[w]:
                tf = word_dic[w][fid]
                idf = self.get_idf(w)
                tf_idf = tf * idf
                if tf_idf > max_tf_idf[0]:
                    max_tf_idf = (tf_idf, tf, idf, fid, w)
        return max_tf_idf
        pass

    def calc_top_n_tf_idf_of_file(self):
        top_n_of_file = {}
        fname_dic = self.fname_dic
        for fname in fname_dic:
            word_list = fname_dic[fname]
            tf_idf_list = []
            for word in word_list:
                w = word.encode("utf-8")
                if w in self.rubbish_set:
                    continue
                if w == "\t":
                    print "going wrong!"
                tf_idf_list.append((self.get_tf_idf_by_file(w, fname), w))
            #top_n_of_file[fname] = heapq.nlargest(int(len(tf_idf_list) * self.proportion) + 1, tf_idf_list)
            top_n_of_file[fname] = heapq.nlargest(1, tf_idf_list)
        return top_n_of_file

    def test_fun(self):
        top_n_of_file = self.calc_top_n_tf_idf_of_file()
        hot_word_dic = {}
        for f in top_n_of_file:
            for word in top_n_of_file[f]:
                if not hot_word_dic.has_key(word[1]):
                    hot_word_dic[word[1]] = 1
                else:
                    hot_word_dic[word[1]] += 1
        return hot_word_dic

    def get_max_tf_idf(self):
        max_l = (0, 0, 0, "")
        word_dic = self.word_dic
        for w in word_dic:
            if w in self.rubbish_set or len(w) < 6:
                continue
            idf = self.get_idf(w)
            total_tf_idf = 0.0
            for f in word_dic[w]:
                total_tf_idf = word_dic[w][f] * idf
            pre_total_tf_idf = total_tf_idf/len(self.fname_dic)
            if pre_total_tf_idf > max_l[0]:
                max_l = (pre_total_tf_idf, idf, w, len(word_dic[w]), word_dic[w])
        return max_l

    def get_top_n_tf_idf(self):
        pass

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

if 0:
    hd = tf_idf("idf.txt", "stopwords.txt")
    rubbish_set = gen_rubbish_set()
    hd.set_rubbish_set(rubbish_set)
    #hd.add_doc(0, "的的的")
    #print hd.get_idf("的")
    #exit()
    #root_path = "/home/kelly/negative_article/01/有争议/"
    root_path = "/home/kelly/combin_article/result/"
    fcounter = 0
    begin = datetime.datetime.now()
    for fname in os.listdir(root_path):
        fcounter += 1
        fpath = os.path.join(root_path, fname)
        with open(fpath, "r") as fd:
            s = fd.read()
            hd.add_doc(fname, s)
    end = datetime.datetime.now()

    print end - begin

    s = hd.dumps()

    with open("dumps.txt", "w") as fd:
        fd.write(s)

    hd = tf_idf()
    hd.loads(s)

    max_tf_idf = hd.get_max_tf_idf()
    print max_tf_idf[0], max_tf_idf[1], max_tf_idf[2], max_tf_idf[3], max_tf_idf[4]

if 0:
    hd = tf_idf()
    rubbish_set = gen_rubbish_set()

    with open("dumps.txt", "r") as fd:
        s =fd.read()
    hd.loads(s)

    word_dic = hd.word_dic
    
    hd.set_rubbish_set(rubbish_set)

    hot_word_dic = hd.test_fun()
    hot_word_list = []
    for word in hot_word_dic:
        hot_word_list.append((hot_word_dic[word], word))
    l = heapq.nlargest(10, hot_word_list)
    for word in l:
        print word[1], word[0]
    exit()
    max_word = (0, "")
    for word in hot_word_dic:
        if hot_word_dic[word] > max_word[0]:
            max_word = (hot_word_dic[word], word)
    print max_word

    #max_tf_idf = hd.get_max_tf_idf()
    #print max_tf_idf[0], max_tf_idf[1], max_tf_idf[2], max_tf_idf[3], max_tf_idf[4]

if 0:
    import heapq

    l = [2, 9, 3, 4, 5, 6, 7, 1, 8]

    heap = heapq.heapify(l)

    print l 

    print heapq.nlargest(3, l)
    print heapq.nsmallest(100, l)

if 0:
    l = range(0, 1000000)

    #print heapq.nlargest(3, l)
    #exit()

    t = top_n(3)

    for i in l:
        t.add_item(i)
    print t.get_sortted_list()

if 0:
    fpath = "/home/kelly/combin_article/result/03_p1_605.txt"
    #fpath = "/home/kelly/tempfile"

    hd = tf_idf()

    with open("dumps.txt", "r") as fd:
        s =fd.read()
    hd.loads(s)
    rubbish_set = gen_rubbish_set()
    
    with open(fpath, "r") as fd:
        s = fd.read()

    tf_idf_list = hd.get_tf_idf_list_by_str(s)
    
    for tf_idf in tf_idf_list:
        print tf_idf[1], tf_idf[0]

if 1:
    #idf生成
    hd = idf()
    
    root_path = "/home/kelly/idf_article/"

    counter = 0
    for fname in os.listdir(root_path):
        print "counter:", counter
        counter += 1
        fpath = os.path.join(root_path, fname)
        with open(fpath, "r") as fd:
            s = fd.read()
        hd.add_doc(s)
    
    s = hd.dumps()
    with open("idf_dumps.txt", "w") as fd:
        fd.write(s)

if 0:
    hd = idf()
    
    with open("idf_dumps.txt", "r") as fd:
        s = fd.read()

    hd.loads(s)
    
    print len(hd.word_dic)

if 0:
    '''
    hot_word整体测试
    '''
    hd = hot_word("/tmp/testdb")

    option_dic = {}
    option_dic["get_word_flag"] = "num"
    option_dic["word_top_num"] = 5

    hd.set_options(option_dic)
    
    root_path = "/home/kelly/negative_article_old/result"
    #root_path = "/home/kelly/negative_test_article/"

    counter = 0
    doc_list = []
    for fname in os.listdir(root_path):
        fpath = os.path.join(root_path, fname)
        with open(fpath, "r") as fd:
            s = fd.read()
        doc_list.append(s)
        #hd.add_doc(fpath)
        #print counter
        counter += 1
    begin = datetime.datetime.now()
    for doc in doc_list:
        hd.add_doc_s(doc)
    end = datetime.datetime.now()

    print end - begin

    print hd.get_top_n_word_list(5)
     
    s = hd.dumps()
        
    with open("hot_word_dumps_5_word.txt", "w") as fd:
        fd.write(s)

if 0:
    with open("hot_word_dumps_5_word.txt", "r") as fd:
        s = fd.read()
    hd = hot_word("/tmp/testdb")

    hd.loads(s)
    
    l = hd.get_top_n_word_list(5)
    
    for i in l:
        print i[0], i[1]

if 0:
    with open("hot_word_dumps.txt", "r") as fd:
        s = fd.read()
    hd = hot_word("/tmp/testdb")

    hd.loads(s)

    s = "我是蓝翔技工拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上总经理，出任CEO，走上人生巅峰。"
    #蓝翔 8.65114949375
    #手扶拖拉机 8.29447454982
    #加薪 7.26485513263
    #技工 6.9081801887
    #升职 6.66523401008

    line_list = []
    res_dic = {}
    with open("/home/kelly/test/sim_test") as fd:
        for l in fd:
            line_list.append(l.strip())
            if len(line_list) >= 3:
                line1 = line_list[0].strip()
                line2 = line_list[1].strip()
                
                l1 = hd.get_file_word_list_by_num(line1, 5)
                l2 = hd.get_file_word_list_by_num(line2, 5)
                
                l1_set = set(l1)
                l2_set = set(l2)
                
                inter1 = (l1_set | l2_set) - l1_set
                inter2 = (l1_set | l2_set) - l2_set
                
                if len(inter1) not in res_dic:
                    res_dic[len(inter1)] = []
                res_dic[len(inter1)].append((line1, line2, len(inter1), len(inter2), l1, l2))
                #print res_dic[len(inter1)][len(res_dic[len(inter1)]) - 1][4]
                #print res_dic[len(inter1)][len(res_dic[len(inter1)]) - 1][5]
                #if len(inter1) > 1 and len(inter2) > 1:
                #    print line1, len(inter1)
                #    print line2, len(inter2)
                #    for i in l1:
                #        print i[1],
                #    print ""
                #    for i in l2:
                #        print i[1],
                #    print ""
                line_list = []
    for i in res_dic[1]:
        print i[0], i[1], i[2], i[3]
        for l in i[4]:
            print l[1],
        print ""
        for l in i[5]:
            print l[1],
        print ""
        print "-" * 80
    for k in res_dic:
        print k, len(res_dic[k])
    exit()
    
    l = hd.get_file_word_list_by_num(s, 5)

    for i in l:
        print i[1], i[0]

    s = "我要去蓝翔，学手扶拖拉机，能够升职加薪，做上技工."
    
    l = hd.get_file_word_list_by_num(s, 5)

    print "-" * 80
    for i in l:
        print i[1], i[0]
    #print "/".join(l)

if 0:
    with open("hot_word_dumps.txt", "r") as fd:
        s = fd.read()

    hd = hot_word()

    hd.loads(s)
    
    with open("/home/kelly/negative_article_old/result/02_n1_192.txt", "r") as fd:
        s = fd.read()
    l = hd.get_file_word_list(s, 10)

    for i in l:
        print i[1], i[0], i

if 0:
    '''
    python对象大小测试
    '''
    import sys

    i = 0

    print "sizeof int:", sys.getsizeof(i)

    s = set()

    print "sizeof empty set:", sys.getsizeof(s)
    for i in range(100000):
        s.add(i)
        if i == 10000:
            print "sizeof 10000 int set", sys.getsizeof(s)
    print "sizeof 100000 int set", sys.getsizeof(s)

if 0:
    dbhd = leveldb.LevelDB("/tmp/testdb")

    dbhd.Put('1', '中文')

    v = dbhd.Get('1')

    print v, type(v)

if 0:
    def test_fun():
        l = [1, 2, 3]
        for i in l:
            yield i

    l = list(test_fun())
    print l

def get_file_tf_idf_list(idf_hd, s):
    word_list = list(jieba.cut(s))
    print len(word_list)
    l_word_dic = {}
    for w in word_list:
        word = w.encode("utf-8")
        if idf_hd.is_rubbish(word):
            continue
        if not l_word_dic.has_key(word):
            l_word_dic[word] = 0
        l_word_dic[word] += 1
    ret_list = []
    print len(l_word_dic)
    for word in l_word_dic:
        tf_idf = l_word_dic[word] * idf_hd.get_idf(word)
        ret_list.append((tf_idf, word))
    return heapq.nlargest(len(ret_list), ret_list)

if 0:
    idf_hd = idf()
    with open("idf_dumps.txt", "r") as fd:
        s = fd.read()
    idf_hd.loads(s)

    with open("/home/kelly/tempfile", "r") as fd:
        s = fd.read()

    #with open("/home/kelly/negative_article_old/result/01_p2_1049.txt", "r") as fd:
    #    s = fd.read()

    for w in get_file_tf_idf_list(idf_hd, s):
        print w[1], w[0]

if 0:
    #word_list = [u'3', u'\u3001', u'\u5386\u53f2', u'\u4e0a', u'\u7684', u'\u4eca\u5929', u'(', u'7', u'\u6708', u'1', u'\u65e5', u'\uff09', u'\r\n', u'\r\n', u'\t', u'\t', u'\t', u'\r\n', u'\uff1f', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'\u4e2d\u56fd\u5171\u4ea7\u515a', u'\u7b2c\u4e00\u6b21', u'\u4ee3\u8868\u5927\u4f1a', u'\u7684', u'\u6700\u540e', u'\u4e00\u5929', u'\u4f1a\u8bae', u'\u662f', u'\u5728', u'\u6d59\u6c5f\u7701', u'\u5609\u5174\u5e02', u'\u5357\u6e56', u'\u4e2d', u'\u4e00\u8258', u'\u753b\u822b', u'\u4e0a', u'\u53ec\u5f00', u'\u7684', u'\u3002', u'\u8fd9', u'\u662f', u'\u6309', u'\u5f53\u5e74', u'\u753b\u822b', u'\u4eff\u9020', u'\u7684', u'\u753b\u822b', u'\uff0c', u'\u505c', u'\u5728', u'\u5357\u6e56', u'\u4e0a\u4f9b', u'\u6e38\u4eba', u'\u77bb\u4ef0', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1921', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd\u5171\u4ea7\u515a', u'\u6210\u7acb', u'\u3002', u'\u8fd9\u662f', u'\u4e2d\u56fd', u'\u5386\u53f2', u'\u4e0a', u'\u5f00\u5929\u8f9f\u5730', u'\u7684', u'\u5927', u'\u4e8b\u4ef6', u'\u3002', u'\u7531\u4e8e', u'\u515a', u'\u7684', u'\u201c', u'\u4e00\u5927', u'\u201d', u'\u53ec\u5f00', u'\u4e8e', u'1921', u'\u5e74', u'7', u'\u6708', u'\uff0c', u'\u800c', u'\u5728', u'\u6218\u4e89', u'\u5e74\u4ee3', u'\u6863\u6848\u8d44\u6599', u'\u96be\u5bfb', u'\uff0c', u'\u5177\u4f53', u'\u5f00\u5e55', u'\u65e5\u671f', u'\u65e0\u6cd5', u'\u67e5\u8bc1', u'\uff0c', u'\u56e0\u6b64', u'\uff0c', u'1941', u'\u5e74', u'6', u'\u6708', u'\u5728', u'\u515a', u'\u6210\u7acb', u'20', u'\u5468\u5e74', u'\u4e4b\u9645', u'\uff0c', u'\u4e2d\u5171\u4e2d\u592e', u'\u53d1\u6587', u'\u6b63\u5f0f', u'\u89c4\u5b9a', u'\uff0c', u'7', u'\u6708', u'1', u'\u65e5\u4e3a', u'\u515a', u'\u7684', u'\u8bde\u751f', u'\u7eaa\u5ff5\u65e5', u'\u3002', u'\u515a', u'\u7684', u'\u4e00\u5927', u'\u5f00\u5e55', u'\u65e5\u671f', u'\u5230', u'20', u'\u4e16\u7eaa', u'70', u'\u5e74\u4ee3', u'\u672b', u'\u624d', u'\u7531', u'\u515a\u53f2', u'\u5de5\u4f5c\u8005', u'\u8003\u8bc1', u'\u6e05\u695a', u'\uff0c', u'\u6839\u636e', u'\u65b0', u'\u53d1\u73b0', u'\u7684', u'\u53f2\u6599', u'\u548c', u'\u8003\u8bc1', u'\u7ed3\u679c', u'\uff0c', u'\u786e\u5b9a', u'\u4e2d\u56fd\u5171\u4ea7\u515a\u7b2c\u4e00\u6b21\u5168\u56fd\u4ee3\u8868\u5927\u4f1a', u'\u4e8e', u'1921', u'\u5e74', u'7', u'\u6708', u'23', u'\u65e5', u'\u5728', u'\u4e0a\u6d77', u'\u6cd5', u'\u79df\u754c', u'\u671b\u5fd7\u8def', u'106', u'\u53f7', u'\u53ec\u5f00', u'\uff0c', u'7', u'\u6708', u'31', u'\u65e5', u'\u6700\u540e', u'\u4e00\u5929', u'\u4f1a\u8bae', u'\u8f6c\u79fb', u'\u5230', u'\u6d59\u6c5f', u'\u5609\u5174', u'\u5357\u6e56', u'\u4e3e\u884c', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1941', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u5171\u4e2d\u592e\u653f\u6cbb\u5c40', u'\u901a\u8fc7', u'\u300a', u'\u5173\u4e8e', u'\u589e\u5f3a\u515a\u6027', u'\u7684', u'\u51b3\u5b9a', u'\u300b', u'\uff0c', u'\u6307\u51fa', u'\uff1a', u'\u8981', u'\u628a', u'\u4e2d\u56fd\u5171\u4ea7\u515a', u'\u8fdb\u4e00\u6b65', u'\u5efa\u8bbe', u'\u6210\u4e3a', u'\u5e7f\u5927', u'\u7fa4\u4f17\u6027', u'\u7684', u'\u3001', u'\u601d\u60f3', u'\u4e0a', u'\u653f\u6cbb', u'\u4e0a', u'\u7ec4\u7ec7', u'\u4e0a', u'\u5b8c\u5168', u'\u5de9\u56fa', u'\u7684', u'\u5e03\u5c14\u4ec0\u7ef4\u514b', u'\u5316', u'\u7684', u'\u515a', u'\uff0c', u'\u4ee5', u'\u62c5\u8d1f\u8d77', u'\u4f1f\u5927', u'\u800c', u'\u8270\u96be', u'\u7684', u'\u9769\u547d', u'\u4e8b\u4e1a', u'\u3002', u'\u8fd9', u'\u5c31', u'\u8981\u6c42', u'\u5168\u4f53', u'\u515a\u5458', u'\u548c', u'\u515a', u'\u7684', u'\u5404\u4e2a', u'\u7ec4\u6210\u90e8\u5206', u'\u90fd', u'\u5728', u'\u7edf\u4e00', u'\u610f\u5fd7', u'\u3001', u'\u7edf\u4e00\u884c\u52a8', u'\u548c', u'\u7edf\u4e00', u'\u7eaa\u5f8b', u'\u4e0b\u9762', u'\uff0c', u'\u56e2\u7ed3\u8d77\u6765', u'\uff0c', u'\u6210\u4e3a', u'\u6709', u'\u7ec4\u7ec7', u'\u7684', u'\u6574\u4f53', u'\uff1b', u'\u8981\u6c42', u'\u5168\u4f53', u'\u515a\u5458', u'\uff0c', u'\u5c24\u5176', u'\u662f', u'\u515a\u5458\u5e72\u90e8', u'\uff0c', u'\u66f4\u52a0', u'\u589e\u5f3a', u'\u81ea\u5df1', u'\u7684', u'\u515a\u6027', u'\u953b\u70bc', u'\uff0c', u'\u628a', u'\u4e2a\u4eba\u5229\u76ca', u'\u670d\u4ece', u'\u4e8e', u'\u515a', u'\u7684', u'\u5229\u76ca', u'\uff0c', u'\u628a', u'\u4e2a\u522b', u'\u515a', u'\u7684', u'\u7ec4\u6210\u90e8\u5206', u'\u7684', u'\u5229\u76ca', u'\u670d\u4ece', u'\u4e8e', u'\u5168\u515a', u'\u7684', u'\u5229\u76ca', u'\uff0c', u'\u4f7f', u'\u5168\u515a', u'\u56e2\u7ed3', u'\u5f97', u'\u50cf', u'\u4e00\u4e2a', u'\u4eba', u'\u4e00\u6837', u'\u3002', u'\u81ea\u6b64\u4ee5\u540e', u'\uff0c', u'\u589e\u5f3a\u515a\u6027', u'\u953b\u70bc', u'\u6210\u4e3a', u'\u515a\u7684\u5efa\u8bbe', u'\u7684', u'\u91cd\u8981', u'\u5185\u5bb9', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\u8fd9\u662f', u'\u5217\u8f66', u'\u5954\u9a70', u'\u5728', u'\u6210\u6606\u94c1\u8def', u'\u4e0a', u'\uff08', u'\u8d44\u6599', u'\u7167\u7247', u'\uff09', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1970', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u5168\u957f', u'1085', u'\u516c\u91cc', u'\u7684', u'\u6210', u'\uff08', u'\u90fd', u'\uff09', u'\u6606', u'\uff08', u'\u660e', u'\uff09', u'\u94c1\u8def', u'\u5efa\u6210', u'\u901a\u8f66', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\u8fd9\u662f', u'\u4e2d\u56fd', u'\u81ea\u5236', u'\u7684', u'\u201c', u'\u97f6\u5c71', u'\u201d', u'\u578b', u'\u7535\u529b\u673a\u8f66', u'\u7275\u5f15', u'\u7740', u'\u5ba2\u8f66', u'\u884c\u9a76', u'\u5728', u'\u5b9d\u6210\u94c1\u8def', u'\u4e0a', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1975', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd', u'\u7b2c\u4e00\u6761', u'\u7535\u6c14\u5316\u94c1\u8def', u'\u2014', u'\u2014', u'\u5b9d', u'\uff08', u'\u9e21', u'\uff09', u'\u6210', u'\uff08', u'\u90fd', u'\uff09', u'\u94c1\u8def', u'\u5efa\u6210', u'\u901a\u8f66', u'\u3002', u'\u5b9d\u6210\u94c1\u8def', u'\u5168\u957f', u'676', u'\u516c\u91cc', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1975', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd', u'\u4e0e', u'\u6cf0\u738b\u56fd', u'\u5efa\u4ea4', u'\u3002', u'\u6cf0\u56fd', u'\u4f4d\u4e8e', u'\u4e9a\u6d32', u'\u4e2d\u5357\u534a\u5c9b', u'\u4e2d\u90e8', u'\uff0c', u'\u9996', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u' ', u'\u90fd', u'\u66fc\u8c37', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\u8fd9\u662f', u'\u5f20\u95fb\u5929', u'\u7684', u'\u8d44\u6599', u'\u7167\u7247', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1976', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd\u5171\u4ea7\u515a', u'\u8457\u540d', u'\u7684', u'\u601d\u60f3\u5bb6', u'\u548c', u'\u653f\u6cbb\u5bb6', u'\u3001', u'\u6770\u51fa', u'\u7684', u'\u65e0\u4ea7\u9636\u7ea7', u'\u9769\u547d\u5bb6', u'\u5f20\u95fb\u5929', u'\u901d\u4e16', u'\uff0c', u'\u4eab\u5e74', u'75', u'\u5c81', u'\u3002', u'\u5f20\u95fb\u5929', u'1925', u'\u5e74', u'\u52a0\u5165', u'\u4e2d\u56fd\u5171\u4ea7\u515a', u'\uff0c', u'1931', u'\u5e74\u4efb', u'\u4e2d\u5171\u4e2d\u592e\u5ba3\u4f20\u90e8', u'\u90e8\u957f', u'\uff0c', u'1934', u'\u5e74\u4efb', u'\u4e2d\u534e\u82cf\u7ef4\u57c3\u5171\u548c\u56fd', u'\u4e2d\u592e', u'\u6267\u59d4\u4f1a', u'\u4eba\u6c11', u'\u59d4\u5458\u4f1a', u'\u4e3b\u5e2d', u'\u3002', u'1935', u'\u5e74', u'1', u'\u6708', u'\u5728', u'\u9075\u4e49\u4f1a\u8bae', u'\u4e0a', u'\u62e5\u62a4', u'\u6bdb\u6cfd\u4e1c', u'\u7684', u'\u9886\u5bfc', u'\uff0c', u'\u4f1a\u540e', u'\u6839\u636e', u'\u4e2d\u592e\u653f\u6cbb\u5c40\u5e38\u59d4', u'\u5206\u5de5', u'\u4ee3\u66ff', u'\u535a\u53e4', u'\u5728', u'\u4e2d\u5171\u4e2d\u592e', u'\u8d1f', u'\u603b\u8d23', u'\u3002', u'1938', u'\u5e74', u'\u540e\u4efb', u'\u4e2d\u5171\u4e2d\u592e\u4e66\u8bb0\u5904', u'\u4e66\u8bb0', u'\u517c', u'\u4e2d\u592e\u5ba3\u4f20\u90e8', u'\u90e8\u957f', u'\u3002', u'\u4ed6', u'\u662f', u'\u4e2d\u5171', u'\u516d\u5c4a', u'\u4e2d\u592e\u653f\u6cbb\u5c40\u5e38\u59d4', u'\u3001', u'\u4e03\u5c4a', u'\u4e2d\u592e\u653f\u6cbb\u5c40', u'\u59d4\u5458', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1980', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u90ae\u7535\u90e8', u'\u51b3\u5b9a', u'\u5728', u'\u5168\u56fd', u'\u8303\u56f4', u'\u5185', u'\u5b9e\u884c', u'\uff1f', u'\u201c', u'\u90ae\u653f\u7f16\u7801', u'\u201d', u'\u5236\u5ea6', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1982', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u5317\u4eac', u'\u65f6\u95f4', u'\u96f6\u65f6', u'\u8d77', u'\uff0c', u'\u7ecf', u'\u56fd\u52a1\u9662', u'\u6279\u51c6', u'\uff0c', u'\u4e2d\u56fd', u'\u7b2c\u4e09\u6b21', u'\u4eba\u53e3\u666e\u67e5', u'\u6b63\u5f0f', u'\u5f00\u59cb', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1984', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd\u94f6\u884c', u'\u5f00\u59cb', u'\u529e\u7406', u'\u4e2d\u56fd', u'\u5883\u5185', u'\u5c45\u6c11', u'\u5b9a\u671f', u'\u5916\u5e01\u5b58\u6b3e', u'\u4e1a\u52a1', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1985', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd', u'\u5f00\u59cb', u'\u5b9e\u884c', u'\u65b0', u'\u7684', u'\u5de5\u8d44\u5236\u5ea6', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1988', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u300a', u'\u6c42\u662f', u'\u300b', u'\u6742\u5fd7', u'\u521b\u520a', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1990', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4ece', u'\u5317\u4eac', u'\u65f6\u95f4', u'\u96f6\u65f6', u'\u8d77\u81f3', u'7', u'\u6708', u'10', u'\u65e5', u'\u5317\u4eac', u'\u65f6\u95f4', u'24', u'\u65f6\u6b62', u'\uff0c', u'\u56fd\u52a1\u9662', u'\u51b3\u5b9a', u'\uff0c', u'\u5728', u'\u4e2d\u56fd', u'\u5927\u9646', u'30', u'\u4e2a\u7701', u'\u3001', u'\u81ea\u6cbb\u533a', u'\u3001', u'\u76f4\u8f96\u5e02', u'\u8fdb\u884c', u'\u4e2d\u56fd', u'\u7b2c\u56db\u6b21', u'\u5168\u56fd', u'\u4eba\u53e3\u666e\u67e5', u'\u767b\u8bb0', u'\u5de5\u4f5c', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1993', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd', u'\u5f00\u59cb', u'\u5b9e\u884c', u'\u65b0', u'\u7684', u'\u8d22\u4f1a', u'\u5236\u5ea6', u'\uff0c', u'\u300a', u'\u4f01\u4e1a\u8d22\u52a1', u'\u901a\u5219', u'\u300b', u'\u300a', u'\u4f01\u4e1a', u'\u4f1a\u8ba1\u51c6\u5219', u'\u300b', u'\u4ee5\u53ca', u'\u5206\u884c\u4e1a', u'\u4f01\u4e1a\u8d22\u52a1', u'\u3001', u'\u4f1a\u8ba1\u5236\u5ea6', u'\u6b63\u5f0f', u'\u751f\u6548', u'\u3002', u'\u8fd9', u'\u662f', u'\u4e3a', u'\u4e2d\u56fd', u'\u8d22\u4f1a', u'\u5236\u5ea6', u'\u4e0e', u'\u56fd\u9645\u60ef\u4f8b', u'\u63a5\u8f68', u'\u800c', u'\u91c7\u53d6', u'\u7684', u'\u91cd\u8981', u'\u4e3e\u63aa', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1997', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd\u653f\u5e9c', u'\u5bf9', u'\u9999\u6e2f', u'\u6062\u590d', u'\u884c\u4f7f\u4e3b\u6743', u'\u3002', u'\u300a', u'\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u9999\u6e2f\u7279\u522b\u884c\u653f\u533a', u'\u57fa\u672c\u6cd5', u'\u300b', u'\u300a', u'\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u9999\u6e2f\u7279\u522b\u884c\u653f\u533a', u'\u9a7b\u519b', u'\u6cd5', u'\u300b', u'\u6b63\u5f0f', u'\u65bd\u884c', u'\u3002', u'\u8fd9\u662f', u'1997', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\u51cc\u6668', u'\uff0c', u'\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u9999\u6e2f\u7279\u522b\u884c\u653f\u533a', u'\u6210\u7acb', u'\u66a8', u'\u7279\u533a\u653f\u5e9c', u'\u5ba3\u8a93\u5c31\u804c', u'\u4eea\u5f0f', u'\u5728', u'\u9999\u6e2f', u'\u4f1a\u8bae\u5c55\u89c8', u'\u4e2d\u5fc3', u'\u65b0\u7ffc', u'\u4e03\u697c', u'\u9686\u91cd\u4e3e\u884c', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'1997', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u4e2d\u56fd\u653f\u5e9c', u'\u5bf9', u'\u9999\u6e2f', u'\u6062\u590d', u'\u884c\u4f7f\u4e3b\u6743', u'\u3002', u'1997', u'\u5e74', u'6', u'\u6708', u'30', u'\u65e5', u'23', u'\u65f6', u'42', u'\u5206', u'\uff0c', u'\u4e2d\u82f1\u4e24\u56fd', u'\u653f\u5e9c', u'\u9999\u6e2f', u'\u653f\u6743', u'\u4ea4\u63a5\u4eea\u5f0f', u'\u5728', u'\u9999\u6e2f', u'\u4e3e\u884c', u'\u3002', u'\u968f\u7740', u'\u82f1\u56fd', u'\u201c', u'\u7c73\u5b57\u65d7', u'\u201d', u'\u56fd\u65d7', u'\u7684', u'\u964d\u4e0b', u'\uff0c', u'\u82f1\u56fd', u'\u5728', u'\u9999\u6e2f', u'\u4e00\u4e2a', u'\u534a\u4e16\u7eaa', u'\u7684', u'\u6b96\u6c11\u7edf\u6cbb', u'\u5ba3\u544a', u'\u7ed3\u675f', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\u706b\u8f66', u'\u5728', u'\u5f53\u96c4', u'\u8349\u539f', u'\u4e0a', u'\u884c\u9a76', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'2006', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u9752\u85cf\u94c1\u8def', u'\u5168\u7ebf', u'\u6b63\u5f0f', u'\u901a\u8f66', u'\u3002', u'\u9752\u85cf\u94c1\u8def', u'\u897f\u5b81', u'\u81f3', u'\u62c9\u8428', u'\u5168\u957f', u'1956', u'\u516c\u91cc', u'\u3002', u'\u5176\u4e2d', u'\uff0c', u'\u897f\u5b81', u'\u81f3', u'\u683c\u5c14\u6728', u'\u6bb5\u4e8e', u'1984', u'\u5e74', u'\u6295\u5165', u'\u8fd0\u8425', u'\u3002', u'2001', u'\u5e74', u'6', u'\u6708', u'\u5f00\u5de5', u'\u4fee\u5efa', u'\u7684', u'\u683c\u5c14\u6728', u'\u81f3', u'\u62c9\u8428', u'\u6bb5', u'\uff0c', u'\u5168\u957f', u'1142', u'\u516c\u91cc', u'\uff0c', u'\u6d77\u62d4', u'4000', u'\u7c73', u'\u4ee5\u4e0a', u'\u7684', u'\u5730\u6bb5', u'\u8fbe', u'960', u'\u516c\u91cc', u'\uff0c', u'\u6700\u9ad8\u70b9', u'\u6d77\u62d4', u'5072', u'\u7c73', u'\uff0c', u'\u7ecf\u8fc7', u'\u8fde\u7eed', u'\u591a\u5e74\u51bb\u571f', u'\u5730\u6bb5', u'550', u'\u516c\u91cc', u'\uff0c', u'\u662f', u'\u4e16\u754c', u'\u94c1\u8def', u'\u5efa\u8bbe\u53f2', u'\u4e0a', u'\u6700\u5177', u'\u6311\u6218\u6027', u'\u7684', u'\u5de5\u7a0b\u9879\u76ee', u'\u3002', u'\u5de5\u7a0b', u'\u7834\u89e3', u'\u4e86', u'\u591a\u5e74\u51bb\u571f', u'\u3001', u'\u9ad8\u5bd2', u'\u7f3a\u6c27', u'\u3001', u'\u751f\u6001', u'\u8106\u5f31', u'\u4e09\u5927', u'\u4e16\u754c\u6027', u'\u5de5\u7a0b', u'\u6280\u672f\u96be\u9898', u'\uff0c', u'\u521b\u9020', u'\u4e86', u'\u591a\u9879', u'\u4e16\u754c', u'\u94c1\u8def', u'\u4e4b', u'\u6700', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'2010', u'\u5e74', u'7', u'\u6708', u'1', u'\u65e5', u'\uff0c', u'\u7ecf', u'\u56fd\u52a1\u9662', u'\u6279\u590d', u'\uff0c', u'\u6df1\u5733\u7ecf\u6d4e\u7279\u533a', u'\u8303\u56f4', u'\u6269\u5927', u'\u5230', u'\u6df1\u5733', u'\u5168\u5e02', u'\u6b63\u5f0f', u'\u5b9e\u65bd', u'\uff0c', u'\u5c06', u'\u5b9d\u5b89', u'\u3001', u'\u9f99\u5c97', u'\u4e24\u533a', u'\u7eb3\u5165', u'\u7279\u533a', u'\u8303\u56f4', u'\uff0c', u'\u8fd9', u'\u610f\u5473\u7740', u'\u4e2d\u56fd', u'\u6700\u65e9', u'\u8bbe\u7acb', u'\u7684', u'\u7ecf\u6d4e\u7279\u533a', u'\u4ece', u'\u539f\u6709', u'\u7684', u'396', u'\u5e73\u65b9\u516c\u91cc', u'\u6269\u5927', u'\u5230', u'1953', u'\u5e73\u65b9\u516c\u91cc', u'\u3002', u'\u6839\u636e', u'\u56fd\u52a1\u9662', u'\u7684', u'\u6279\u590d', u'\uff0c', u'\u7279\u533a', u'\u8303\u56f4', u'\u6269\u5927', u'\u540e', u'\uff0c', u'\u73b0\u6709', u'\u7684', u'\u7279\u533a', u'\u7ba1\u7406', u'\u7ebf', u'\u6682\u65f6', u'\u4fdd\u7559', u'\uff0c', u'\u4e0d\u518d', u'\u65b0', u'\u8bbe', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'\u8fd9\u5f20', u'\u7167\u7247', u'\u663e\u793a', u'\u7684', u'\u662f', u'\u6fb3\u5927\u5229\u4e9a', u'\u6089\u5c3c\u5e02', u'\u666f\u8272', u'\uff08', u'4', u'\u6708', u'13', u'\u65e5\u6444', u'\uff09', u'\u3002', u'\u767d\u8272', u'\u5efa\u7b51', u'\u4e3a', u'\u6089\u5c3c\u6b4c\u5267\u9662', u'\u3002', u'\r\n', u'\r\n', u'\uff1f', u'\uff1f', u'\uff1f', u'\uff1f', u'7', u'\u6708', u'1', u'\u65e5', u'\u662f', u'\u4e16\u754c', u'\u5efa\u7b51', u'\u8282', u'\u3002', u'\u4e16\u754c', u'\u5efa\u7b51', u'\u8282\u662f', u'1985', u'\u5e74', u'6', u'\u6708', u'\u56fd\u9645', u'\u5efa\u7b51\u5e08', u'\u534f\u4f1a', u'\u7b2c', u'63', u'\u6b21', u'\u7406\u4e8b\u4f1a', u'\u4e0a', u'\u786e\u7acb', u'\u7684', u'\uff0c', u'\u4ee5\u540e', u'\u53c8', u'\u8fdb\u4e00\u6b65', u'\u89c4\u5b9a', u'\uff0c', u'\u4eca\u540e', u'\u6bcf\u5e74', u'\u7684', u'\u4e16\u754c', u'\u5efa\u7b51', u'\u8282', u'\u5c06', u'\u540c', u'\u8054\u5408\u56fd', u'\u53f7\u53ec', u'\u7684', u'\u56fd\u9645', u'\u5e74', u'\u4e3b\u9898', u'\u7ed3\u5408', u'\u8d77\u6765', u'\u5f00\u5c55', u'\u6d3b\u52a8', u'\u3002', u'\r\n', u'\r\n', u'\r\n', u'\uff1f', u'\r\n', u'\r\n', u'\r\n', u'\r\n', u'\u4e0a\u6d77', u'\u5efa\u515a', u'\uff0c', u'\u5f00\u5929\u8f9f\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u5357\u660c', u'\u5efa\u519b', u'\uff0c', u'\u60ca\u5929\u52a8\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u745e\u91d1', u'\u5efa\u653f', u'\uff0c', u'\u7ffb\u5929\u8986\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u5317\u4eac', u'\u5efa\u56fd', u'\uff0c', u'\u6539\u5929\u6362\u5730', u'\u3002', u'\r\n', u'\r\n', u'\u65b0\u578b', u'\u4e2d\u534e', u'\uff0c', u'\u7ecf\u5929\u7eac\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u6297\u7f8e', u'\u6838\u7206', u'\uff0c', u'\u611f\u5929\u52a8\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u548c\u5e73', u'\u5d1b\u8d77', u'\uff0c', u'\u9876\u5929\u7acb\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u91d1\u878d\u5371\u673a', u'\uff0c', u'\u4e0a\u5929\u5165\u5730', u'\uff1b', u'\r\n', u'\r\n', u'\u4e2d\u534e', u'\u590d\u5174', u'\uff0c', u'\u62dc\u5929', u'\u8c22\u5730', u'\u3002', u'\r\n', u'\t', u'\t', u'\t', u'\t', u'\t', u'\t', u'\t', u'\r\n', u'\t', u'\t', u'\r\n']
    
    s = ''.join(word_list)
    
    print s

if 0:
    s = "测试.......测试......."

    l = list(jieba.cut(s))

    print "/".join(l)

if 0:
    import redis

    rhd = redis.Redis()
    
    root_path = "/home/kelly/negative_article_old/result"

    doc_list = []
    for fname in os.listdir(root_path):
        fpath = os.path.join(root_path, fname)
        with open(fpath, "r") as fd:
            s = fd.read()
        doc_list.append(s)

    begin = datetime.datetime.now()
    counter = 0
    for doc in doc_list:
        rhd.set(counter, doc)
        counter += 1
    end = datetime.datetime.now()

    print end - begin

if 0:
    dbhd = leveldb.LevelDB("/tmp/testdb")

    root_path = "/home/kelly/negative_article_old/result"

    counter = 0
    doc_list = []
    for fname in os.listdir(root_path):
        fpath = os.path.join(root_path, fname)
        with open(fpath, "r") as fd:
            s = fd.read()
        doc_list.append(s)
        counter += 1

    begin = datetime.datetime.now()
    counter = 0
    for doc in doc_list:
        dbhd.Put(str(counter), doc)
        counter += 1
    end = datetime.datetime.now()
    
    print end - begin

if 0:
    #测试leveldb 的 writeBatch
    #批量写入 比 单条写入快很多倍
    import sys
    dbhd = leveldb.LevelDB("/tmp/testdb")
    batch = leveldb.WriteBatch()
    print len(batch)

    root_path = "/home/kelly/negative_article_old/result"

    counter = 0
    doc_list = []
    for fname in os.listdir(root_path):
        fpath = os.path.join(root_path, fname)
        with open(fpath, "r") as fd:
            s = fd.read()
        doc_list.append(s)
        counter += 1

    begin = datetime.datetime.now()
    counter = 0
    for doc in doc_list:
        batch.Put(str(counter), doc)
        counter += 1
    dbhd.Write(batch)
    print len(batch)
    end = datetime.datetime.now()
    
    print end - begin

if 0:
    url_re = re.compile(r'(http:\/\/)*[\w\d]+\.[\w\d\.]+\/[\w\d_!@#$%^&\*-_=\+]+')
    s = "顺德两男子路边摘几个芒果 被警方拘留5日_大粤网_腾讯网 http://url.cn/PPSLio"
    s = '武汉今日起上调社保缴费基数_房产武汉站_腾讯网 http://url.cn/IvaXt2'
    
    print re.sub(url_re, "", s)
    
    print url_re.sub("", s)

if 0:
    l = [(1, 'a'), (2, 'b'), (3, 'c')]

    t = [j[1] for j in l]

    print t
