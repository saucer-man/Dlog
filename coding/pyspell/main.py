#!/usr/bin/env python

import sys
import spell as s

if __name__ == '__main__':
    slm = s.lcsmap('[\\s]+')
    with open("./train.log", 'r', encoding='utf-8') as f:
        for i in f.readlines():  # sys.stdin.readlines():
            sub = i.strip('\n')
            obj = slm.insert(sub)
            # print(obj.get_id(), obj.param(sub))

    # 打印出结果，postion的意思是*的下标位置
    print("处理好的日志为")
    slm.dump()
    # 0 {"lcsseq": "this is * pen ", "lineids": [1, 2, 3], "postion": [2]}
    # 1 {"lcsseq": "i am * ", "lineids": [4, 5], "postion": [2]}
    # 2 {"lcsseq": "i am * and * ", "lineids": [6, 7], "postion": [2, 4]}

    # 将训练结果保存在这里
    s.save('test.pickle', slm)

    # 加载保存好的结果
    slm = s.load('test.pickle')
    #slm.dump()
    with open("./test.log", 'r', encoding='utf-8') as f:
        print("训练的结果为：")
        for i in f.readlines():
            sub = i.strip('\n')
            obj = slm.match(sub)
            print(obj.tojson())

