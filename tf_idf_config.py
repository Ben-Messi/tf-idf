#!/usr/bin/env python
# -*- coding: utf-8 -*-
dict_path = "/home/kelly/code/git/tf-idf/jieba.dict.utf8"
hmm_path = "/home/kelly/code/git/tf-idf/hmm_model.utf8"
idf_dumps_path = "/home/kelly/code/git/tf-idf/idf.txt"
stopwords_path = "/home/kelly/code/git/tf-idf/stopwords.txt"
short_url_path = "/home/kelly/code/git/tf-idf/short_url.txt"
default_idf = 13.6
#min_interval = 600
#max_interval = 2 * 24 * 3600
min_interval = 1
max_interval = 3
word_list_len = 5
redis_host = "192.168.2.97"
redis_port = 6379
redis_db = 0
min_interval_key = "min_interval"
max_interval_key = "max_interval"
