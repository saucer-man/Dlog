import keras
import numpy as np
from nltk.probability import FreqDist
from nltk.util import ngrams
from lib.log_key_lstm_model import log_key_model
import os
from lib.common import save, load

# get the sequence with 3:1
def standard_log_key(log_key_sequence_str):
    # 将日志键， 通过滑动窗口分为一个一个日志序列，这里将其分为4个日志键为一个序列
    tokens = log_key_sequence_str.split(' ')
    # 将日志键其变为int
    tokens = [int(i) for i in tokens]
    K = max(tokens)+1  # 日志键的种类个数
    # print("the tokens are:",tokens)
    bigramfdist_4 = FreqDist()
    bigrams_4 = ngrams(tokens, 4)
    # from nltk.util import ngrams
    # a = ['1', '2', '3', '4', '5']
    # b = ngrams(a, 2)
    # for i in b:
    #     print
    #     i
    # ('1', '2')
    # ('2', '3')
    # ('3', '4')
    # ('4', '5')
    bigramfdist_4.update(bigrams_4)
    print("the bigramfdsit_4 is:", list(bigramfdist_4.keys()))
    # we set the length of history logs as 3
    seq = np.array(list(bigramfdist_4.keys()))

    # print("the seq is:",seq)
    X, Y = seq[:, :3], seq[:, 3:4]
    # print(seq.shape)   # (253, 4)
    # print(X_normal.shape)  # (253, 3)
    # print(Y_normal.shape)  # (253, 1)
    X = np.reshape(X, (-1, 3, 1))
    # print(X_normal)
    # [[[6]
    #   [72]
    #   [6]]
    #
    #  [[72]
    #     [6]
    #     [6]]
    #  ...]
    # 将数字等比缩小，变为从0到1
    X = X / K
    # 将整型标签转为onehot
    num_classes = len(list(set(Y.T.tolist()[0]))) + 1 # num_classes指的是Y_normal的种类
    Y = keras.utils.to_categorical(Y)   # num_classes=num_classes
    return X, Y


def execution_path(df_train_log, df_test_log):
    # 提取normal日志键执行流
    train_log_key_sequence_str = " ".join([str(EventId) for EventId in df_train_log["EventId"]])
    # 将日志流通过滑动窗口分为4个日志为一个日志序列
    X_train, Y_train = standard_log_key(train_log_key_sequence_str)

    # 对日志序列进行lstm训练和测试
    lg = log_key_model()
    execution_path_model_filepath = "tmpdata/ExecutePathModel/model.pkl"
    if os.path.isfile(execution_path_model_filepath):
        model = load(execution_path_model_filepath)
    else:
        model = lg.train(X_train, Y_train)
        save(execution_path_model_filepath, model)


    # 提取test日志键执行流
    test_log_key_sequence_str = " ".join([str(EventId) for EventId in df_test_log["EventId"]])
    # 将日志流通过滑动窗口分为4个日志为一个日志序列
    X_test, Y_test = standard_log_key(test_log_key_sequence_str)

    anomaly_sequence = lg.predcit(model, X_test, Y_test)
    print('the length of anomaly_sequence {} is {}'.format(anomaly_sequence, len(anomaly_sequence)))

    # anomaly_sequence为异常的序列
    anormal_lineid_list = [i+3 for i in anomaly_sequence]
    df_anormal = df_test_log.loc[df_test_log["LineId"].isin(anormal_lineid_list)]
    df_anormal.to_csv("tmpdata/execute_path_anormal.csv", index=False)