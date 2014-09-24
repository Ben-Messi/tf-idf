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

class repeat_filter():
    def __init__(self, idf_path, stop_words_path = ""):
        self.tf_idf_hd = tf_idf(idf_path, stop_words_path)
        self.fid = 0
        self.repeat = 0
        self.not_repeat = 1
        self.r_hd = redis.Redis()
        self.word_key_pre = "word:"
        self.title_id_pre = "title_id:"
        self.time_limit = 259200

        self.r_hd.flushdb()

    def insert_s_to_redis(self, s, word_list):
        '''
        传入的word_list已添加前缀
        '''
        fid = self.fid
        self.fid += 1
        r_hd = self.r_hd

        p = r_hd.pipeline()
        for word in word_list:
            p.sadd(word, fid)
            p.expire(word, self.time_limit)
        p.set(self.title_id_pre + str(fid), s)
        p.expire(self.title_id_pre + str(fid), self.time_limit)
        p.execute()

    def filter(self, s):
        """
        重复返回 0
        不重复返回 1
        """
        word_list = self.tf_idf_hd.get_top_n_tf_idf(s)

        print "/".join(word_list)

        key_word_list = [self.word_key_pre + word for word in word_list]

        for i in range(len(word_list)):
            key_word_list = [self.word_key_pre + word for word in word_list]
            del key_word_list[i]
            fid_set = self.r_hd.sinter(key_word_list)

            if fid_set:
                return self.repeat
        self.insert_s_to_redis(s, key_word_list)
        return self.not_repeat

if 1:
    flter = repeat_filter("idf.txt", "stopwords.txt")

    title = "湖北英山县委常委陆海华涉嫌违纪违法被调查[图]"

    ret = flter.filter(title)

    print ret

    title = "湖北英山县委常委陆海华涉嫌违纪违法被调查"
    ret = flter.filter(title)

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

