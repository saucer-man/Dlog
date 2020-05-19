#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

class lcsobj():
    # 初始化参数
    def __init__(self, objid, seq, lineid, refmt):
        # 0 ['this', 'is', 'a', 'pen'] 1 [\s]+
        self._refmt = refmt   # 这个是分隔符，空格 \n \r之类的
        if isinstance(seq, str) == True:
            self._lcsseq = re.split(self._refmt, seq.lstrip().rstrip())
        else:
            self._lcsseq = seq  # 日志键的list模式
        self._lineids = [lineid]
        self._pos = []   # _pos就是lcs列表中值为*的下标
        self._sep = "	"
        self._id = objid
        return

    # return seq和self._lcsseq相同元素的个数，用来计算最长公共子序列
    def getlcs(self, seq):
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        count = 0
        lastmatch = -1
        for i in range(len(self._lcsseq)):
            #if self._lcsseq[i] == '*':
            if self._ispos(i) == True:
                continue
            for j in range(lastmatch+1, len(seq)):
                if self._lcsseq[i] == seq[j]:
                    lastmatch = j
                    count += 1
                    break
        return count

    # 将lineid插入到self._lineids
    def insert(self, seq, lineid):
        # print(seq, lineid)
        # ['this', 'is', 'the', 'pen'] 2
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        self._lineids.append(lineid)
        temp = ""
        lastmatch = -1
        placeholder = False  # placeholder代表是否已经有*，如果没有*并且不匹配则添加一个*
        # 为了防止this is a pen 和 this is the bigger pen 产生 this is ** pen

        # 底下是为了将变量替换成*
        # 比如self._lcsseq原本是['this', 'is', 'a', 'pen']
        # 现在要插进来一个['this', 'is', 'the', 'pen'] 2
        # 那么我就把self._lcsseq变成['this', 'is', '*', 'pen']
        for i in range(len(self._lcsseq)):
            #下面等同于if self._lcsseq[i] == '*':
            if self._ispos(i) == True:
                if not placeholder:
                    temp = temp + "* "
                placeholder = True
                continue
            for j in range(lastmatch+1, len(seq)):
                if self._lcsseq[i] == seq[j]:
                    placeholder = False
                    temp = temp + self._lcsseq[i] + " "
                    lastmatch = j
                    break
                elif not placeholder:
                    temp = temp + "* "
                    placeholder = True
        temp = temp.lstrip().rstrip()
        self._lcsseq = re.split(" ", temp)
        
        self._pos = self._get_pos()
        self._sep = self._get_sep()

    # 将obj转换为json格式返回
    def tojson(self, seq):

        temp = ""
        for i in self._lcsseq:
            temp = temp + i + " "
        ret = {}
        ret["lcsseq"] = temp
        ret["lineids"] = self._lineids
        ret["postion"] = self._pos
        ret["lcsseq_id"] = self._id
        ret["param"] = self.param(seq)
        return ret

    # 返回seq的长度
    def length(self):
        return len(self._lcsseq)

    # 返回seq在obj中的变量部分
    # 比如seq为 this is a pen
    # self._lcsseq为 this is a *
    # 那么变量部分就为["pen"]
    def param(self, seq):
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        j = 0
        ret = []
        for i in range(len(self._lcsseq)):
            slot = []
            if self._ispos(i) == True:
                while j < len(seq):
                    if i != len(self._lcsseq)-1 and self._lcsseq[i+1] == seq[j]:
                        break
                    else:
                        slot.append(seq[j])
                    j += 1
                ret.append(slot)
            else:
                j += 1
        return ret

    def re_param(self, seq):
        if isinstance(seq, list) == True:
            seq = ' '.join(seq)
        seq = seq.lstrip().rstrip()

        ret = []
        print(self._sep)
        print(seq)
        p = re.split(self._sep, seq)
        for i in p:
            if len(i) != 0:
                ret.append(re.split(self._refmt, i.lstrip().rstrip()))
        if len(ret) == len(self._pos):
            return ret
        else:
            return []


    # 判断下标为idx的列表元素是否为*
    def _ispos(self, idx):
        for i in self._pos:
            if i == idx:
                return True
        return False

    def _tcat(self, seq, s, e):
        sub = ''
        for i in range(s, e + 1):
            sub += seq[i] + " "
        return sub.rstrip()

    def _get_sep(self):
        sep_token = []
        s = 0
        e = 0
        for i in range(len(self._lcsseq)):
            if self._ispos(i) == True:
                if s != e:
                    sep_token.append(self._tcat(self._lcsseq, s, e))
                s = i + 1
                e = i + 1
            else:
                e = i
            if e == len(self._lcsseq) - 1:
                sep_token.append(self._tcat(self._lcsseq, s, e))
                break

        ret = ""
        for i in range(len(sep_token)):
            if i == len(sep_token)-1:
                ret += sep_token[i]
            else:
                ret += sep_token[i] + '|'
        return ret

    def _get_pos(self):
        pos = []
        for i in range(len(self._lcsseq)):
            if self._lcsseq[i] == '*':
                pos.append(i)
        return pos

    # 返回obj的id，也就是在map列表中的下标
    def get_id(self):
        return self._id

class lcsmap():

    def __init__(self, refmt):
        self._refmt = refmt
        self._lcsobjs = []
        self._lineid = 0
        self._id = 0
        return

    def insert(self, entry):
        seq = re.split(self._refmt, entry.lstrip().rstrip())

        # seq代表将其分割之后的列表
        # this is a pen
        # --> ['this', 'is', 'a', 'pen']

        obj = self.match(seq)
        # 这个是去看是否匹配
        # 如果不匹配，则添加新的obj到_lcsobjs列表中
        if obj == None:
            self._lineid += 1
        #   print(self._id, seq, self._lineid, self._refmt)
        #   0 ['this', 'is', 'a', 'pen'] 1 [\s]+
            obj = lcsobj(self._id, seq, self._lineid, self._refmt)
            self._lcsobjs.append(obj)
            self._id += 1
        # 如果匹配，则修改已有的obj，插入一个lineid
        else:
            self._lineid += 1
            obj.insert(seq, self._lineid)

        return obj

    # 判断seq是否已经存在在obj列表中，基于LCS
    def match(self, seq):
        # seq = ['this', 'is', 'a', 'pen']
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        bestmatch = None
        bestmatch_len = 0
        seqlen = len(seq)

        # 遍历每个obj，看下seq是否match
        for obj in self._lcsobjs:
            objlen = obj.length()
            if objlen < seqlen/2 or objlen > seqlen*2: continue

            l = obj.getlcs(seq)
            # l代表seq和obj中相同的元素个数

            if l >= seqlen/2 and l > bestmatch_len:
                bestmatch = obj
                bestmatch_len = l
        # print(bestmatch)
        return bestmatch

    def objat(self, idx):
        return self._lcsobjs[idx]

    def size(self):
        return len(self._lcsobjs)
    # 将map列表打印出来
    def dump(self):
        count = 0
        for i in self._lcsobjs:
            print(count, i.tojson())
            count += 1


