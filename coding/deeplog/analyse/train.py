import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import os
import pickle
import time

def generate(name, window_size):
    num_sessions = 0 # num_sessions 表示总共多少行，也就是多少个任务流
    inputs = []
    outputs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            # 这里将日志键-1，保证了日志键从0开始
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))

    return inputs, outputs


def lstm_train(x, y, num_epochs, batch_size):
    # according to the article, we will stack two layers of LSTM, the model about stacked LSTM for sequence classification
    model = Sequential()
    # =============== model 1 ===================
    # input data shape: (batch_size, timesteps, data_dim)
    # 首先确定输入的特征维度 return_sequence=True 表示多对多
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
    # 防止过拟合
    # model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))  # 这里必须是false  https://stackoverflow.com/questions/51763983/error-when-checking-target-expected-dense-1-to-have-3-dimensions-but-got-array
    #model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(64, return_sequences=False))
    # output layer with a single value prediction (1,K) 输出激活函数的
    model.add(Dense(y.shape[1], activation='softmax'))
    # 定义损失函数 损失函数loss，优化器optimizer，评估指标metrics
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    # to ensure the training data patterns remain sequential --- disable the shuffle
    # 定义checkpoint
    # filepath = "weights/log_weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # make it stateful, we add batch_size 确定连接权重
    model.fit(x, y, epochs=num_epochs, batch_size=batch_size, shuffle=True)   # callbacks=[checkpoint]
    # to see the summary of input and output shape 输出各层的形状
    # model.summary()
    # print('the accuracy for single lstm model is:', model.evaluate(x, y, batch_size=batch_size, verbose=0))
    return model


if __name__ == "__main__":
    start_time = time.time()

    num_classes = 28  # 日志键总类
    num_epochs = 50  # 迭代次数，需要找到一个最好的迭代次数
    batch_size = 2048  # 决定下降的方向 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小
    window_size = 10  # 每个序列的日志总数

    TP = 0  # 判定为异常实际上为异常
    FP = 0  # 判定为异常实际上为正常
    n_candidates = 10  # top n probability of the next tag


    print("获取训练数据")
    X, Y = generate('hdfs_train', window_size)
    # print(inputs[0:3])
    # print(outputs[0:3])
    print("对数据进行一定处理")
    #  reshape X to be [samples, time steps, features]
    # print(X[0]) (4, 4, 4, 21, 10, 8, 10, 8, 10, 8)
    # 形如（samples，timesteps，input_dim）的3D张量
    X = np.reshape(X, (len(X), window_size, 1))  # 将横的变成竖的
    X = X / float(num_classes)
    # print(Y[0])  # 25
    Y = np_utils.to_categorical(Y, num_classes)  # 转成one hot,维度为总的tags数量
    # print(Y[0])  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    model = lstm_train(X, Y, num_epochs, batch_size)
    # 将对象以二进制形式保存

    # with open("model", 'wb') as f:
    #     pickle.dump(model, f)

    # 训练完之后，选择一个loss最小的进行train
    # module_name_list = os.listdir("weights")
    # epoch = {}
    # import re
    #
    # for module in module_name_list:
    #     epoch_num, loss = re.findall(r"log_weights-improvement-(.+?)-(.+?)-bigger.hdf5", module)[0]
    #     epoch[epoch_num] = loss
    #
    # print(epoch)
    def generate1(name, window_size):
        # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
        # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
        # hdfs = set()
        hdfs = []
        with open('data/' + name, 'r') as f:
            for ln in f.readlines():
                # 还是将所有的值都-1
                ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
                # 如果当前数组不足windows_size，则将其后面补上-1
                ln = ln + [-1] * (window_size + 1 - len(ln))
                # hdfs.add(tuple(ln))
                hdfs.append(tuple(ln))
        print('Number of sessions({}): {}'.format(name, len(hdfs)))
        return hdfs


    test_normal_loader = generate1('hdfs_test_normal', window_size)
    test_abnormal_loader = generate1('hdfs_test_abnormal', window_size)

    for line in test_abnormal_loader:
        for i in range(len(line) - window_size):
            seq = line[i: i + window_size]
            label = line[i + window_size]
            X = np.reshape(seq, (1, window_size, 1))  # 将横的变成竖的
            X = X / float(num_classes)
            Y = np_utils.to_categorical(label, num_classes)  # 转成one hot,维度为总的tags数量
            prediction = model.predict(X, verbose=0)  # 输出一个len(tags)的向量，数值越高的列对应概率最高的类别
            if np.argmax(Y) not in prediction.argsort()[0][::-1][: n_candidates]:
                # 如果实际值不在预测值的前三概率，则系统判定为异常日志，此时
                TP += 1
                break



    for line in test_normal_loader:
        for i in range(len(line) - window_size):
            seq = line[i:i + window_size]
            label = line[i + window_size]
            X = np.reshape(seq, (1, window_size, 1))  # 将横的变成竖的
            X = X / float(num_classes)
            Y = np_utils.to_categorical(label, num_classes)  # 转成one hot,维度为总的tags数量

            prediction = model.predict(X, verbose=0)  # 输出一个len(tags)的向量，数值越高的列对应概率最高的类别
            if np.argmax(Y) not in prediction.argsort()[0][::-1][: n_candidates]:
                # 如果实际值不在预测值的前三概率，则系统判定为异常日志，此时
                FP += 1
                break

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))


    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    TN = len(test_normal_loader) - FP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

    print(f"FP:{FP}")
    print(f"FN: {FN}")
    print(f"TP: {TP}")
    print(f"TN: {TN}")

    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
