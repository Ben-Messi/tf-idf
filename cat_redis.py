#!/usr/bin/python
# -*- coding: UTF-8 -*-

import redis

if 1:
    r_hd = redis.Redis()

    keys = r_hd.keys("*")

    print keys

    for k in keys:
        try:
            ret = r_hd.smembers(k)
        except:
            ret = r_hd.get(k)
        print k, ret
