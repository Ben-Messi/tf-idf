#!/usr/bin/python
# -*- coding: UTF-8 -*-

import redis
import datetime
import random
import pickle
import os
import time
import jieba
import ujson
import heapq
import copy
import fast_search
import hot_word
from similarity import similarity

from tf_idf_test import tf_idf, idf

def read_file(filename):
    fd = file(filename, "r")
    title = fd.readline()
    title = title.strip()
    content = fd.read()
    fd.close()
    return (title, content)

class pub_repeat_filter():
    def __init__(self, idf_path, stop_words_path = "", r_hd = 0):
        self.tf_idf_hd = tf_idf(idf_path, stop_words_path)
        self.repeat = 0
        self.not_repeat = 1
        if not r_hd:
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
    def __init__(self, idf_path, stop_words_path = "", uid_overtime_path = ""):
        self.uid_overtime_dic = self.gen_uid_overtime_dic(uid_overtime_path)
        self.tf_idf_hd = tf_idf(idf_path, stop_words_path)
        self.repeat = 0
        self.not_repeat = 1
        #如果title中topN小于此值则将结果判断相似度
        self.sim_judge_limit = 3
        self.r_hd = redis.Redis()
        self.word_key_pre = "main_word:"
        self.title_id_pre = "main_title_id:"
        self.uid_pre = "main_tid_uid:"
        self.time_stamp_pre = "main_time_stamp:"
        self.time_limit = 259200
        self.uid_overtime_default = self.time_limit 
        self.main_title_id_key = "incr:main_title_id"
     
        self.r_hd.flushdb()
    
    def gen_uid_overtime_dic(self, uid_overtime_path):
        ret_dic = {}
        if not uid_overtime_path:
            return ret_dic
        with open(uid_overtime_path) as fd:
            for l in fd:
                idx = l.find("\t")
                uid = int(l[:idx])
                overtime = int(l[idx + 1:].strip())
                ret_dic[uid] = overtime
        return ret_dic

    def get_max_time_limit(self, id_set):
        max_time = 0
        for uid in id_set:
            tmp = self.uid_overtime_dic.get(uid, 0)
            if tmp > max_time:
                max_time = tmp
        return max_time

    def get_main_title_id(self):
        main_title_id = self.r_hd.incr(self.main_title_id_key)
        return int(main_title_id)

    def insert_s_to_redis(self, s, word_list, id_set):
        '''
        s         : 不带前缀的title完整字符串
        word_list : title中提取的高权重词, 带前缀
        id_set    : 需要预警的uid
        '''
        if not id_set:
            return 
        max_ttl = self.get_max_time_limit(id_set)
        tid = self.get_main_title_id()
        tid_key = self.title_id_pre + str(tid)
        uid_tid_key = self.uid_pre + str(tid)
        p = self.r_hd.pipeline()
        print "word_list", word_list
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
        self.r_hd.set(self.time_stamp_pre + str(tid), int(time.time()))
        self.r_hd.expire(self.time_stamp_pre + str(tid), max_ttl)
    
    def update_uid_to_redis(self, repeat_tid_list, word_list, id_set):
        '''
        repeat_tid : 重复的所有title id
        word_list  : title中提取的高权重词, 带前缀
        id_set     : 需要预警的uid
        '''
        if not id_set:
            return 
        max_ttl = self.get_max_time_limit(id_set)
        for tid in repeat_tid_list:
            tid_key = self.title_id_pre + str(tid)
            uid_tid_key = self.uid_pre + str(tid)
            p = self.r_hd.pipeline()
            #word:
            #仅更新ttl
            for word in word_list:
                p.ttl(word)
            ret_list = p.execute()
            p = self.r_hd.pipeline()
            #word ttl
            for i in range(len(word_list)):
                ttl = ret_list[i] if ret_list[i] > max_ttl else max_ttl
                p.expire(word_list[i], ttl)
            #tid:
            p.expire(tid_key, max_ttl)
            #uid_tid:
            p.sadd(uid_tid_key, *id_set)
            p.execute()
            ttl = self.r_hd.ttl(uid_tid_key)
            ttl = ttl if ttl > max_ttl else max_ttl
            self.r_hd.expire(uid_tid_key, ttl)
    
    def check_for_uid_overtime(self, tid_set, uid_set):
        overtime_uid_set = set()
        cur_time = int(time.time())
        for tid in tid_set:
            time_stamp_key = self.time_stamp_pre + str(tid)
            time_stamp = int(self.r_hd.get(time_stamp_key))

            interval = cur_time - time_stamp
            for uid in uid_set:
                overtime = self.uid_overtime_dic.get(uid, 0)
                print overtime, interval
                if overtime < interval:
                    overtime_uid_set.add(uid)
        return overtime_uid_set

    def filter(self, s, id_set):
        """
        返回需要预警的id_set
        """
        word_list = self.tf_idf_hd.get_top_n_tf_idf(s)
        word_list_len = len(word_list)
        
        print "/".join(word_list)
        
        repeat_tid_set = set()
        ret_set = id_set
        for i in range(word_list_len):
            key_word_list = [self.word_key_pre + word for word in word_list]
            if word_list_len > 1:
                del key_word_list[i]
            tid_set_s = self.r_hd.sinter(key_word_list)
            tid_set = set([int(i) for i in tid_set_s])
            #fid_set为重复的id集合, 加到总重复id集合里
            repeat_tid_set |= tid_set

        key_word_list = [self.word_key_pre + word for word in word_list]
        if repeat_tid_set:
            repeat_tid_list = list(repeat_tid_set)
            if word_list_len < self.sim_judge_limit:
                title_key_list = [self.title_id_pre + str(i) for i in repeat_tid_list]
                #取出所有title判断相似度
                title_list = self.r_hd.mget(title_key_list)
                idx = -1
                for title in title_list:
                    idx += 1
                    if similarity(s, title) > 0.5:
                        break
                if idx >= 0:
                    l_id_set_s = self.r_hd.smembers(self.uid_pre + str(repeat_tid_list[idx]))
                    l_id_set = set([int(i) for i in l_id_set_s])
                    overtime_uid_set = self.check_for_uid_overtime(repeat_tid_set, id_set)
                    ret_set = (id_set - l_id_set) | overtime_uid_set
                    self.update_uid_to_redis(repeat_tid_list, key_word_list, ret_set)
                else:
                    #如果没有相似的title, 则insert
                    self.insert_s_to_redis(s, key_word_list, ret_set)
            else:
                tid_uid_key_list = [self.uid_pre + str(tid) for tid in repeat_tid_set]
                l_id_set_s = self.r_hd.sunion(tid_uid_key_list)
                l_id_set = set([int(i) for i in l_id_set_s])
                overtime_uid_set = self.check_for_uid_overtime(repeat_tid_set, id_set)
                print "overtime uid set:", overtime_uid_set
                print "repeat tid set:", repeat_tid_set
                ret_set = (id_set - l_id_set) | overtime_uid_set
                self.update_uid_to_redis(repeat_tid_list, key_word_list, ret_set)
        else:
            #如果一个都没有对上　则直接新增
            self.insert_s_to_redis(s, key_word_list, id_set)
        return ret_set

def test_fun():
    r_hd = redis.Redis()
    r_hd.flushdb()

    #l = ['main_word:a', 'main_word:b', 'main_word:c']
    l = ['main_word:a']

    ret = r_hd.sunion(l)
    
    print ret

if 1:
    #class test
    class test_c:
        def __init__(self, v):
            self.t = v

        def __isub__(self, tc):
            self.t -= tc.t

    l = range(100000)
    counter = 100
    d = {}
    for i in l:
        d[i] = i
    a = 0
    begin = datetime.datetime.now()
    for c in range(counter):
        for i in l:
            a = 0
            a += i
    end = datetime.datetime.now()

    print end - begin

    begin = datetime.datetime.now()
    for c in range(counter):
        for i in d:
            a = 0
            a += i
    end = datetime.datetime.now()

    print end - begin

if 0:
    e = hot_word.hot_event("test")
    with open("/home/kelly/tempfile") as fd:
        s = fd.read()
    while 1:
        e.add_doc_s(s)
        

if 0:
    #循环list测试
    l = hot_word.loop_list()
    l2 = list(l)
    exit()
    i = iter(l)
    i.next()
    i.next()
    print len(l)

if 0:
    #复杂度测试
    counter = 100000
    begin = datetime.datetime.now()
    for i in range(counter):
        l = [1] * 8000
        del l[7999]
    end = datetime.datetime.now()

    print end - begin

    begin = datetime.datetime.now()
    for i in range(counter):
        l = [1] * 8000
        del l[0]
    end = datetime.datetime.now()

    print end - begin

if 0:
    seg_hd = hot_word.cppjieba()
    seg_hd.initialize()
    seg_hd.cut("中华人民共和国")

if 0:
    h = hot_word.hot_word()
    #p = "/home/kelly/negative_article_old/result/01_p2_983.txt"
    #title, content = read_file(p)
    #h.add_doc_s(title + content)

    #exit()

    root_dir = "/home/kelly/negative_article_old/result"
    article = []
    counter = 0
    for fname in os.listdir(root_dir):
        counter += 1
        #if counter > 10000:
        #    break
        title, content = read_file(os.path.join(root_dir, fname))
        json_data = {}
        #l = list(jieba.cut(title + content))
        #for i in range(len(l)):
        #    l[i] = l[i].encode("utf-8")
        #json_data["jieba_cut"] = l
        json_data["title"] = title
        json_data["content"] = content
        article.append(json_data)

    #jieba.initialize()
    #begin = datetime.datetime.now()
    #word_list = []
    ##counter = 0
    #for i in [] and article:
    #    l = list(jieba.cut(i["title"] + i["content"]))
    #    #for j in range(len(l)):
    #    #    l[j] = l[j].encode("utf-8")
    #    word_list.append(l)
    #end = datetime.datetime.now()
    #print end - begin
    #print "append done."
    #print "article len:", len(article)
    begin = datetime.datetime.now()
    #counter = 0
    for i in article:
        #print counter
        #counter += 1
        h.add_doc_s(i["title"] + i["content"])
    end = datetime.datetime.now()

    print end - begin

    l = h.get_top_n_word_list(3)

    print "/".join([i[1] for i in l])

if 0:
    idf_jieba_path = "/home/kelly/code/warning/key/idf.txt"
    idf_new_path = "/home/kelly/code/git/tf-idf/idf.txt"
    idf_full_path = "/home/kelly/code/codebackup/idf_full.txt"

    new_idf = {}
    with open(idf_new_path, "r") as fd:
        for l in fd:
            idx = l.find("\t")
            w = l[:idx]
            idf = float(l[idx + 1:])
            new_idf[w] = idf
    jieba_idf = {}
    with open(idf_jieba_path, "r") as fd:
        for l in fd:
            idx = l.find("\t")
            w = l[:idx]
            idf = float(l[idx + 1:])
            jieba_idf[w] = idf
    full_idf = new_idf
    full_idf.update(jieba_idf)

    full_idf_list = []
    for w in full_idf:
        full_idf_list.append((full_idf[w], w))

    full_idf_list = heapq.nlargest(len(full_idf_list), full_idf_list)

    with open(idf_full_path, "w") as fd:
        for w in full_idf_list:
            fd.write(w[1] + "\t" + str(w[0]) + "\n")
    exit()

    diff_counter = 0
    same_counter = 0
    for w in jieba_idf:
        if w not in new_idf:
            diff_counter += 1
        else:
            same_counter += 1
    print same_counter, diff_counter, len(jieba_idf), len(new_idf)

    diff_counter = 0
    same_counter = 0
    for w in new_idf:
        if w not in jieba_idf:
            diff_counter += 1
        else:
            same_counter += 1
    print same_counter, diff_counter, len(jieba_idf), len(new_idf)

if 0:
    idf_hd = idf()
    with open("idf_dumps.txt") as fd:
        s = fd.read()
        idf_hd.loads(s)
    idf_hd.save_idf_by_fpath("/home/kelly/backup/idf.txt")

if 0:
    from coverage import coverage
    cov = coverage()     #生成coverage对象
    cov.start()         #开始分析

    flter = main_repeat_filter("idf.txt", "stopwords.txt", "uid_overtime.txt")
    #s = "a, b, c, d, e"
    s = "x"
    id_set = set([1, 2, 3])

    ret = flter.filter(s, id_set)

    print ret
    time.sleep(2)
    #raw_input(">>")

    #s = "a, b, c, d, f"
    s = "x"
    id_set = set([2, 3, 4, 5])

    ret = flter.filter(s, id_set)

    #-------------------------------------------------------
    s = "a, b, c, d, e"
    id_set = set([1, 2, 3])

    ret = flter.filter(s, id_set)

    print ret
    time.sleep(2)
    #raw_input(">>")

    s = "a, b, c, d, f"
    id_set = set([2, 3, 4, 5])

    ret = flter.filter(s, id_set)
    print ret
    cov.stop()            #分析结束
    cov.save()            #将覆盖率结果保存到数据文件

if 0:
    tf_idf_hd = tf_idf("idf.txt", "stopwords.txt")
    
    s = "a, b, c, d, e, f, 1, 2, 3"
    print tf_idf_hd.get_top_n_tf_idf(s)

if 0:
    main_flter = main_repeat_filter("idf.txt", "stopwords.txt")
    main_flter.insert_s_to_redis('abc', ['main_word:a', 'main_word:b', 'main_word:c'], set([10, 20, 30]))

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

