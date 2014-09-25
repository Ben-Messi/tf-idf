#!/usr/bin/python
# -*- coding: UTF-8 -*-

import redis
import datetime
import random
import pickle
import os
import jieba
import ujson
import heapq
import copy
import fast_search

from tf_idf_test import tf_idf, idf

class pub_repeat_filter():
    def __init__(self, idf_path, stop_words_path = ""):
        self.tf_idf_hd = tf_idf(idf_path, stop_words_path)
        self.repeat = 0
        self.not_repeat = 1
        self.r_hd = redis.Redis()
        self.word_key_pre = "pub_word:"
        self.title_id_pre = "pub_title_id:"
        self.time_limit = 259200
        self.pub_title_id_key = "incr:pub_title_id"

        self.r_hd.flushdb()

    def insert_s_to_redis(self, s, word_list):
        '''
        传入的word_list已添加前缀
        '''
        tid = self.get_pub_title_id()
        r_hd = self.r_hd

        p = r_hd.pipeline()
        for word in word_list:
            p.sadd(word, tid)
            p.expire(word, self.time_limit)
        tid_key = self.title_id_pre + str(tid)
        p.set(tid_key, s)
        p.expire(tid_key, self.time_limit)
        p.execute()

    def get_pub_title_id(self):
        pub_title_id = self.r_hd.incr(self.pub_title_id_key)
        return int(pub_title_id)

    def filter(self, s):
        """
        重复返回 0
        不重复返回 1
        """
        word_list = self.tf_idf_hd.get_top_n_tf_idf(s)

        print "/".join(word_list)

        for i in range(len(word_list)):
            key_word_list = [self.word_key_pre + word for word in word_list]
            del key_word_list[i]
            tid_set = self.r_hd.siner(key_word_list)

            if tid_set:
                return self.repeat
        self.insert_s_to_redis(s, key_word_list)
        return self.not_repeat

class main_repeat_filter():
    def __init__(self, idf_path, stop_words_path = ""):
        self.tf_idf_hd = tf_idf(idf_path, stop_words_path)
        self.repeat = 0
        self.not_repeat = 1
        self.r_hd = redis.Redis()
        self.word_key_pre = "main_word:"
        self.title_id_pre = "main_title_id:"
        self.uid_pre = "main_tid_uid:"
        self.time_stamp = "main_time_stamp:"
        self.time_limit = 259200
        self.main_title_id_key = "incr:main_title_id"
     
        self.r_hd.flushdb()
    
    def get_max_time_limit(id_set):
        return 259200

    def get_main_title_id(self):
        main_title_id = self.r_hd.incr(self.main_title_id_key)
        return int(main_title_id)

    def insert_s_to_redis(self, s, word_list, id_set):
        '''
        s         : 不带前缀的title完整字符串
        word_list : title中提取的高权重词, 带前缀
        id_set    : 需要预警的id
        '''
        if not id_set:
            return 
        max_ttl = self.get_max_time_limit()
        tid = self.get_main_title_id()
        tid_key = self.title_id_pre + str(tid)
        uid_tid_key = self.uid_pre + str(tid)
        p = self.r_hd.pipeline()
        #word:
        for word in word_list:
            p.sadd(word, tid)
            p.ttl(word)
        ret_list = p.execute()
        p = self.r_hd.pipeline()
        #word ttl
        for i in range(len(word_list)):
            ttl = ret_list[i * 2 + 1] if ret_list[i * 2 + 1] > max_ttl else max_ttl
            p.expire(word_list[i], ttl)
        #tid:
        p.set(tid_key, s)
        p.expire(tid_key, max_ttl)
        #uid_tid:
        p.sadd(uid_tid_key, *id_set)
        p.execute()
        ttl = self.r_hd.ttl(uid_tid_key)
        ttl = ttl if ttl > max_ttl else max_ttl
        self.r_hd.expire(uid_tid_key, ttl)
        self.r_hd.set(self.time_stamp + str(tid), int(time.time()))
        self.r_hd.expire(self.time_stamp + str(tid), max_ttl)
    
    def filter(self, s, id_set):
        """
        返回需要预警的id_set
        """
        word_list = self.tf_idf_hd.get_top_n_tf_idf(s)
        
        print "/".join(word_list)
        
        repeat_tid_set = set()
        for i in range(len(word_list)):
            key_word_list = [self.word_key_pre + word for word in word_list]
            del key_word_list[i]
            tid_set_s = self.r_hd.sinter(key_word_list)
            tid_set = set([int(i) for i in tid_set_s])
            #fid_set为重复的id集合, 加到总重复id集合里
            repeat_tid_set |= tid_set
        l_id_set_s = self.r_hd.sunion([self.uid_pre + str(tid) for tid in repeat_tid_set])
        l_id_set = set([int(i) for i in l_id_set_s])
        left_set = id_set - l_id_set 
        self.insert_s_to_redis(s, key_word_list, left_set)
        return left_set

def test_fun():
    r_hd = redis.Redis()
    r_hd.flushdb()

    #l = ['main_word:a', 'main_word:b', 'main_word:c']
    l = ['main_word:a']

    ret = r_hd.sinter(l)
    
    print ret

if 0:
    main_flter = main_repeat_filter("idf.txt", "stopwords.txt")
    main_flter.insert_s_to_redis('abc', ['main_word:a', 'main_word:b', 'main_word:c'], set([10, 20, 30]))

if 1:
    test_fun()

if 0:
    pub_flter = pub_repeat_filter("idf.txt", "stopwords.txt")

    title = "广西一官员获刑十年未坐一天牢 法院称系监外执行"
    
    ret = pub_flter.filter(title)
    
    print ret

    title = "广西贪官获刑十年未坐一天牢 法院称系监外执行"
    ret = pub_flter.filter(title)

    print ret

if 0:
    hd = tf_idf("idf.txt", "stopwords.txt")
    def repeat_filter(s):
        l = hd.get_top_n_tf_idf(s)

if 0:
    with open("/home/kelly/tempfile") as fd:
        s = fd.read()
    hd = tf_idf("idf.txt", "stopwords.txt")

    word_list = hd.get_top_n_tf_idf(s)
    print "/".join(word_list)
    pass

if 0:
    hd = idf()
    with open("idf_dumps.txt") as fd:
        s = fd.read()
    
    hd.loads(s)
    
    max_idf = [(0, "")]
    for w in hd.word_dic:
        l_idf = hd.get_idf(w)
        if l_idf > max_idf[0][0]:
            max_idf = [(l_idf, w)]
        elif l_idf == max_idf[0][0]:
            max_idf.append((l_idf, w))
    
    for w_t in max_idf:
        print w_t[1], w_t[0]
    print len(max_idf)
    print len(hd.word_dic)

