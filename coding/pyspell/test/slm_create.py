#!/usr/bin/env python

import sys
sys.path.append("../")

import spell as s


if __name__ == '__main__':
    slm = s.lcsmap('[\\s]+') 
    # \s匹配任何空白字符，包括空格、制表符、换页符等等
    f = open("./train.log", 'r', encoding='utf-8')
    for i in f.readlines():
        sub = i.strip('\n')
        obj = slm.insert(sub)
        # print(obj.get_id(), obj.param(sub))
    s.save('slm.pickle', slm)
    f.close()

slm.dump()
