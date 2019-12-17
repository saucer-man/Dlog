#!/usr/bin/env python

import sys
sys.path.append("../")

import spell as s

if __name__ == '__main__':
    slm = s.load('slm.pickle')
    #slm.dump()
    for i in sys.stdin.readlines():
        sub = i.strip('\n')
        obj = slm.match(sub)
        print(obj.tojson)
