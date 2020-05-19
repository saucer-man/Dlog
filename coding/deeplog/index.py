# coding=utf-8
# /usr/bin/env python3

import sys
from lib import spell
import re
import os

import pandas as pd
from lib.common import save, load, time_elapsed
from lib.execution_path_detect import execution_path
from lib.param_value_detect import param_value
sys.setrecursionlimit(2000)


def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(r"\\ +", r'\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    # regex = "(?P<Month>.*?) (?P<Date>.*?) (?P<Time>.*?) (?P<Type>.*?) (?P<Component>.*?): (?P<Content>.*?)"
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers, logformat):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]

                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def get_content(log_file, logformat):
    # 从日志文件中分理出单纯的日志
    headers, regex = generate_logformat_regex(logformat)
    # print(headers) # ['Month', 'Date', 'Time', 'Type', 'Component', 'Content']
    # print(regex) # re.compile('^(?P<Month>.*?) (?P<Date>.*?) (?P<Time>.*?) (?P<Type>.*?) (?P<Component>.*?): (?P<Content>.*?)$')
    df_log = log_to_dataframe(log_file, regex, headers, logformat)
    return df_log

def spell_log(df_log, df_type="train"):
    spell_result_path = "tmpdata/SpellResult/spell.pkl"
    if os.path.isfile(spell_result_path):
        slm = load(spell_result_path)
        # 加载保存好的结果
    else:
        # 首先需要训练一遍，找出所有的日志健，保存在spell_result_path中
        # 要选取可以涵盖所有日志类型的数据用来训练
        slm = spell.lcsmap('[\\s]+')
        for i in range(len(df_log)):
            log_message = df_log["Content"][i]
            # print(log_message)
            sub = log_message.strip('\n')
            slm.insert(sub)
        # 将spell的训练结果保存在这里
        save(spell_result_path, slm)

    # 对每条日志进行训练一遍，然后保存在spell_result.txt中
    templates = [0] * df_log.shape[0]
    ids = [0] * df_log.shape[0]
    ParameterList = [0] * df_log.shape[0]
    time_interval = [0] * df_log.shape[0]
    for i in range(len(df_log)):
        log_message = df_log["Content"][i].strip()
        obj = slm.insert(log_message)
        # seq = re.split('[\\s]+', log_message)
        # ParameterList[i] = obj.param(seq) # 取出log中的参数
        # if param != []:
        #     param = reduce(operator.add, param)  # 多维数组变一维数组
        obj_json = obj.tojson(log_message)
        templates[i] = obj_json["lcsseq"]  # 获取该日志条目的日志键
        ids[i] = obj_json["lcsseq_id"]  # 获取日志键id 也就是事件编号
        ParameterList[i] = obj_json["param"]  # 取出log中的参数

    # 生成两个日志时间差，加入param参数中
    # print(df_log.shape)
    # print(len(df_log))
    for id in range(len(df_log)):
        if id == 0:
            time_interval[id] = "0"
        else:
            time_last = df_log["Time"][id-1]
            time_now = df_log["Time"][id]
            elapsed = time_elapsed(time_last, time_now)
            time_interval[id] = elapsed
        ParameterList[id].append(time_interval[id])

    # 将结果保存在df_log里面
    df_log['EventId'] = ids # 事件向量
    df_log['EventTemplate'] = templates  # 日志模板 日志键
    df_log["ParameterList"] = ParameterList

    df_log.to_csv(f"tmpdata/struct/{df_type}_structured.csv", index=False)
    return df_log



if __name__ == '__main__':
    logformat = '<Month> <Date> <Time> <Type> <Component>: <Content>' # Linux ssh都适用
    # 定义一些数据路径
    train_log_file = "logdata/Linux/train.log"
    test_log_file = "logdata/Linux/train.log"

    # 创建中间文件的存放目录
    tmpdata_path = ["struct", "EventNpy", "SpellResult", "ParamData", "ParamModel", "ExecutePathModel"]
    for path in tmpdata_path:
        if not os.path.exists(f"tmpdata/{path}"):
            os.makedirs(f"tmpdata/{path}")

    # 提取train
    print("提取train数据")
    df_train_log = get_content(train_log_file, logformat)  # 读取到的log
    df_train_log = spell_log(df_train_log, df_type='train')  # 做日志模板提取，参数提取之后的df_log

    # 提取test
    print("提取test数据")
    df_test_log = get_content(test_log_file, logformat)  # 读取到的log
    df_test_log = spell_log(df_test_log, df_type='test')  # 做日志模板提取，参数提取之后的df_log

    # 执行路径异常检测
    print("执行路径异常检测")
    execution_path(df_train_log, df_test_log)

    # 参数值向量异常检测
    print("参数值向量异常检测")
    param_value(df_train_log, df_test_log)