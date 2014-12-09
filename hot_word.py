#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import datetime
import random
import pickle
from bitarray import bitarray
import re
import os
import threading
import jieba
import redis
import math
import ujson
import cPickle
import heapq
import copy
import leveldb
import fast_search
import c_tf_idf_tk
import tf_idf_config

class cppjieba:
    def __init__(self, dict_path = "jieba.dict.utf8", hmm_path = "hmm_model.utf8"):
        self.dict_path = dict_path
        self.hmm_path = hmm_path
        self.inited = 0
        self.seg_hd = 0

    def initialize(self):
        if not self.inited:
            self.seg_hd = c_tf_idf_tk.cppjieba_init(self.dict_path, self.hmm_path)
        self.inited = 1

    def cut(self, s):
        if not self.inited:
            self.initialize()
        return c_tf_idf_tk.cppjieba_cut(self.seg_hd, s)
    
    def __del__(self):
        c_tf_idf_tk.cppjieba_close(self.seg_hd)

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

class train_idf:
    def __init__(self, dics = {}):
        self.word_dic = dics
        self.fcounter = 0
        self.default_idf = 10
        self.log_base = math.e
        self.rubbish_set, self.rubbish_hd = self.get_rubbish_set()
        cur_path = os.path.dirname(__file__)
        dict_path = tf_idf_config.dict_path
        hmm_path = tf_idf_config.hmm_path
        self.seg_hd = cppjieba(dict_path, hmm_path)

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
        word_set = set(self.seg_hd.cut(s))
        for word in word_set:
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

class idf:
    def __init__(self):
        self.idf_dic = {}
        self.stopword_set = set()
        self.default_idf = tf_idf_config.default_idf

    def load_idf(self, idf_path):
        with open(idf_path) as fd:
            for l in fd:
                idx = l.find("\t")
                if idx <= 0:
                    continue
                word = l[:idx]
                if not word:
                    continue
                if l[idx + 1:]:
                    idf_v = float(l[idx + 1:])
                else:
                    idf_v = 0
                self.idf_dic[word] = idf_v

    def load_stopwords(self, stopword_path):
        with open(stopword_path) as fd:
            for l in fd:
                self.stopword_set.add(l.strip())
        self.stopword_set.add(" ")
        self.stopword_set.add("\t")
        self.stopword_set.add("\n")
        self.stopword_set.add("\r")
        self.stopword_set.add("\r\n")

    def load(self, idf_path, stopword_path):
        self.load_idf(idf_path)
        self.load_stopwords(stopword_path)
    
    def get_idf(self, w):
        return self.idf_dic.get(w, self.default_idf)
    
    def is_rubbish(self, w):
        return w in self.stopword_set

class hot_word:
    idf_hd = None
    seg_hd = None
    short_url_hd = None
    url_re = None
    def __init__(self):
        if not hot_word.idf_hd:
            hot_word.idf_hd = idf()
            hot_word.idf_hd.load(tf_idf_config.idf_dumps_path, tf_idf_config.stopwords_path)
        if not hot_word.seg_hd:
            hot_word.seg_hd = cppjieba(tf_idf_config.dict_path, tf_idf_config.hmm_path)
    
        if not hot_word.short_url_hd:
            hot_word.short_url_hd = fast_search.load(tf_idf_config.short_url_path)
        if not hot_word.url_re:
            hot_word.url_re = re.compile(r'(http:\/\/)*[\w\d]+\.[\w\d\.]+\/[\w\d_!@#$%^&\*-_=\+]+')
        self.hot_word_dic = {}
        self.get_file_word_flag = "num"
        self.word_list_n = 5
        self.get_file_word_cbk = {}
        self.get_file_word_cbk["num"] = self.get_file_word_list_by_num
        self.get_file_word_cbk["percent"] = self.get_file_word_list_by_persent

    def clear(self):
        self.hot_word_dic.clear()

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
        if self.get_file_word_flag == "percent":
            self.word_list_n = self.word_list_persent
        elif self.get_file_word_flag == "num":
            self.word_list_n = self.word_list_num
        else:
            self.word_list_n = 0

    def dumps(self):
        dics = {}
        dics["hot_word_dic"] = pickle.dumps(self.hot_word_dic)
        self.write_and_clean_batch()
        s = pickle.dumps(dics)
        return s
    
    def loads(self, s):
        dics = pickle.loads(s)
        hot_word.idf_hd = idf()
        hot_word.idf_hd.load(tf_idf_config.idf_path, tf_idf_config.stopwords_path)
        self.hot_word_dic = pickle.loads(dics["hot_word_dic"])

    def __isub__(self, hw_hd):
        d = hw_hd.hot_word_dic
        for word in d:
            self.hot_word_dic[word] = self.hot_word_dic.get(word, 0)
            self.hot_word_dic[word] -= d[word]
            if not self.hot_word_dic[word]:
                del self.hot_word_dic[word]
        return self

    def __iadd__(self, hw_hd):
        d = hw_hd.hot_word_dic
        for word in d:
            self.hot_word_dic[word] = self.hot_word_dic.get(word, 0)
            self.hot_word_dic[word] += d[word]

    def get_file_word_list_by_num(self, s, n):
        '''
        获取文章tf-idf值top_n的词列表
        '''
        word_list = hot_word.seg_hd.cut(s)

        return self.get_file_word_list_base(word_list, n)

    def get_file_word_list_by_persent(self, s, n):
        '''
        获取文章tf-idf值top n%的词列表
        '''
        word_list = hot_word.seg_hd.cut(s)
        #使用去重后的词数
        word_num = int(len(set(word_list)) * n * 0.01)
        if not word_num:
            word_num = 1
        #print "%d:%d " % (word_num, len(word_list))

        return self.get_file_word_list_base(word_list, word_num)

    #@profile
    def get_file_word_list_base(self, word_list, n):
        '''
        传入分好的文章词列表
        '''
        #l = word_list[:n]
        #ret = []
        #for i in l:
        #    ret.append((1, i))
        #return ret
        l_word_dic = {}
        for word in word_list:
            if hot_word.idf_hd.is_rubbish(word):
                continue
            if not l_word_dic.has_key(word):
                l_word_dic[word] = 0
            l_word_dic[word] += 1
        ret_list = []
        for word in l_word_dic:
            tf_idf = l_word_dic[word] * hot_word.idf_hd.get_idf(word)
            ret_list.append((tf_idf, word))
        l = heapq.nlargest(n, ret_list)
        return l

    def s_filter(self, s):
        result = fast_search.findall(hot_word.short_url_hd, s)
        if result:
            s = hot_word.url_re.sub("", s)
        return s

    #此doc不再加到idf中, idf不再变化
    def add_doc(self, fname):
        try:
            with open(fname, "r") as fd:
                s = fd.read()
        except:
            return -1
        
        self.add_doc_s(s)
    
    #@profile
    def add_doc_word_list(self, word_list):
        ret_list = self.get_file_word_list_base(word_list, self.word_list_n)
        self.add_word_to_dic(ret_list)

    def get_word_list(self, s):
        s = self.s_filter(s)
        word_list = self.seg_hd.cut(s)
        return word_list

    def add_doc_s(self, s):
        s = self.s_filter(s)
        word_list = self.get_file_word_cbk[self.get_file_word_flag](s, self.word_list_n)
        self.add_word_to_dic(word_list)

    #@profile
    def add_word_to_dic(self, word_list):
        for w in word_list:
            try:
                word = w[1]
            except:
                exit()
            if not self.hot_word_dic.has_key(word):
                self.hot_word_dic[word] = 0
            self.hot_word_dic[word] += 1

    #不带频数的word_list
    def get_top_n_word_list(self, n):
        l = self.get_top_n_word_list_with_freq(n)
        return [i[1] for i in l]

    #带频数的word_list
    def get_top_n_word_list_with_freq(self, n):
        hot_word_list = []
        for word in self.hot_word_dic:
            hot_word_list.append((self.hot_word_dic[word], word))
        l = heapq.nlargest(n, hot_word_list)
        return l

#无需外界管理数组头
#只需append,loop_list类会管理长度及数组头
#__init__(self, 长度, 类型)
class loop_list:
    def __init__(self, list_len, T):
        self.T = T
        self.is_full = 0
        self.list_len = list_len
        self.cur_idx = 0
        self.head_idx = 0
        self.l = []
        for i in range(list_len):
            self.l.append(T())

    def __getitem__(self, idx):
        if idx >= self.list_len or idx < 0:
            raise IndexError
        else:
            return self.l[(self.head_idx + idx) % self.list_len]

    def __setitem__(self, idx, v):
        if idx >= self.list_len or idx < 0:
            raise IndexError
        else:
            self.l[(self.head_idx + idx) % self.list_len] = v

    def __len__(self):
        return self.list_len

    def move_next(self):
        '''
        cur_idx指向下一个
        '''
        tmp_idx = (self.cur_idx + 1) % self.list_len
        if tmp_idx < self.cur_idx:
            self.is_full = 1
        self.cur_idx = tmp_idx

        if self.is_full:
            self.head_idx = (self.head_idx + 1) % self.list_len

        if hasattr(self.l[self.cur_idx], 'clear'):
            self.l[self.cur_idx].clear()
        else:
            self.l[self.cur_idx] = self.T()

    def to_list(self):
        ret_list = []
        i = 0
        while ((self.head_idx + i) % self.list_len) != self.cur_idx:
            ret_list.append(self.__getitem__(i))
            i += 1

    def end(self):
        '''
        返回最后一个元素的索引, 外部使用
        '''
        if self.is_full:
            return self.list_len - 1
        else:
            return self.cur_idx

    def __del__(self):
        pass

#内部使用循环数组做hot_word对象管理
#todo: 如果性能过低,尝试使用同时加到min,max里,如果min过期,则从max里减去min
class hot_event:
    def __init__(self, pre_fix):
        self.pre_fix = pre_fix
        self.init_time = time.time()
        self.cur_time = self.init_time
        self.min_interval = tf_idf_config.min_interval
        self.max_interval = tf_idf_config.max_interval
        self.max_counter = 0
        #此处实例化hot_word类后类属性会初始化,loop_list创建对象则不会再费时间
        self.max_hd = hot_word()
        self.hd_list = loop_list(int(self.max_interval / self.min_interval), hot_word)
        self.rhd = redis.Redis(host = tf_idf_config.redis_host, port = tf_idf_config.redis_port, db = tf_idf_config.redis_db)


        self.thread_lock = threading.Lock()
        t = threading.Thread(target=self.refresh, args=())
        t.setDaemon(True)
        t.start()

    def refresh(self):
        while 1:
            t = time.time()
            self.thread_lock.acquire()
            self.max_counter += 1
            self.store_word_lists()
            if self.hd_list.is_full:
                self.max_hd -= self.hd_list[0]
            self.hd_list.move_next()
            self.cur_time = t
            self.thread_lock.release()
            time.sleep(tf_idf_config.min_interval)

    def add_doc_s(self, s):
        word_list = self.max_hd.get_word_list(s)
        self.thread_lock.acquire()
        self.hd_list[self.hd_list.end()].add_doc_word_list(word_list)
        self.max_hd.add_doc_word_list(word_list)
        self.thread_lock.release()

    def get_top_n_word_list_with_freq(self, n):
        '''
        返回tuple
        元素1为当前min_interval中热门词
        元素2为当前max_interval中热门词
        '''
        min_word_list = self.hd_list[self.hd_list.end()].get_top_n_word_list_with_freq(n)
        max_word_list = self.max_hd.get_top_n_word_list_with_freq(n)
        return (min_word_list, max_word_list)

    def get_top_n_word_list(self, n):
        '''
        返回tuple
        元素1为当前min_interval中热门词
        元素2为当前max_interval中热门词
        '''
        freq_word_lists = self.get_top_n_word_list_with_freq(n)
        return ([i[1] for i in freq_word_lists[0]], [i[1] for i in freq_word_lists[1]])

    def store_word_lists(self):
        '''
        生成min,max word_list, 并存储
        '''
        word_lists = self.get_top_n_word_list(tf_idf_config.word_list_len)
        min_key = "%s:%s" % (self.pre_fix, tf_idf_config.min_interval_key)

        dt = datetime.datetime.now()
        min_json = {}
        min_json["words"] = word_lists[0]
        min_json['ctime'] = dt
        min_json_str = ujson.dumps(min_json)
        self.rhd.rpush(min_key, min_json_str)
        if (self.max_interval / self.min_interval) <= self.max_counter:
            self.max_counter = 0
            max_key = "%s:%s" % (self.pre_fix, tf_idf_config.max_interval_key)
            max_json = {}
            max_json["words"] = word_lists[1]
            max_json['ctime'] = dt
            max_json_str = ujson.dumps(max_json)
            self.rhd.rpush(max_key, max_json_str)

    def __del__(self):
        pass
