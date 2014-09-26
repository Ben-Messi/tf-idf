#!/usr/bin/python
# -*- coding: UTF-8 -*-
import c_distance_compare

def similarity(base, s):
    #global sim_counter
    #sim_counter += 1
    if len(base) > len(s):
        base, s = s, base
    distance = distance_compare(base, s)
    if len(base) <= 0:
        return 0

    if distance >= len(base):
        distance = len(base)

    return 1.0 - (distance * 1.0 / len(base))

def distance_compare(s, t):
    #if len(s) > len(t):
    #    s, t = t, s
    #第一步
    n = len(s)
    m = len(t)

    #如果s长度不及t的一半 直接认为不相似
    #可节省一些无谓的比较时间
    if n < m/2:
        return n

    return c_distance_compare.distance_compare(s, t, n, m)


if __name__ == "__main__":
    pass
