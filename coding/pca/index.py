import pandas as pd
from collections import OrderedDict
import re
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from pca import PCA

class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`
            oov: bool, whether to use OOV event
            min_count: int, the minimal occurrence of events (default 0), only valid when oov=True.

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        self.term_weighting = term_weighting  # tf-idf
        self.normalization = normalization  # zero-mean
        self.oov = oov  # false

        X_counts = []
        for i in range(X_seq.shape[0]):
            # 将每个日志的每个事件计数组成event_counts
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        # print(X_seq)
        # [list(['E9']) list(['E5', 'E5']) list(['E5', 'E22', 'E5', 'E5', 'E11'])]
        # print(X_counts)
        # [Counter({'E9': 1}), Counter({'E5': 2}), Counter({'E5': 3, 'E22': 1, 'E11': 1})]
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0) # 填充空值
        # print(X_df)
        #        E9   E5  E22  E11
        #     0  1.0  0.0  0.0  0.0
        #     1  0.0  2.0  0.0  0.0
        #     2  0.0  3.0  1.0  1.0
        
        self.events = X_df.columns  # 所有的event事件
        X = X_df.values   # 上面的矩阵 3行四列


        num_instance, num_event = X.shape # 行数 列数 3，4


        # TF-IDF模型  Term Frequency Inverse Document Frequency
        # TF-IDF用以评估一字词对于一个语料库中的其中一份文件的重要程度
        # http://www.ruanyifeng.com/blog/2013/03/tf-idf.html
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0) # 列相加
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        # 经过pf-idf模型
        # [[2. 0. 0. 0.]        [[ 0.13515503 -0.36620409 -0.36620409 -0.36620409]
        # [0. 1. 0. 0.]    -->   [-0.67577517  0.73240819 -0.36620409 -0.36620409]
        # [3. 0. 1. 1.]]        [ 0.54062014 -0.36620409  0.73240819  0.73240819]]

        # 标准化，数据符合标准正态分布，即均值为0，标准差为1
        # print(X)
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        # [[0.81093021 0.         0.         0.        ]    
        #  [0.         1.09861228 0.         0.        ]  
        #  [1.21639531 0.         1.09861228 1.09861228]]
        # --> 
        # [[ 0.13515503 -0.36620409 -0.36620409 -0.36620409]
        #  [-0.67577517  0.73240819 -0.36620409 -0.36620409]
        #  [ 0.54062014 -0.36620409  0.73240819  0.73240819]]
        X_new = X
        
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 
        return X_new

    def transform(self, X_seq):
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        if self.oov:
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 

        return X_new


def load_HDFS(log_file, train_ratio=0.5):
    print("读取log数据")
    # 读取log csv文件到struct_log
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)

    # 字典类型，会自动根据key排序
    data_dict = OrderedDict()

    # 遍历读取到的日志
    for idx, row in struct_log.iterrows():
        # 读取Content列中的blk字段，
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            # 以blk_id为键，eventid列表为值，组成一个字典
            data_dict[blk_Id].append(row['EventId'])

    # 将字典转化为DataFrame类型
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

    # 保存到csv中
    data_df.to_csv('data_instances.csv', index=False)

    # 将其分割为两个部分，x_train 和x_test 根据train_ratio的比例来分割
    x_data = data_df['EventSequence'].values
    num_train = int(train_ratio * x_data.shape[0])
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]

    # 将train和test数据打乱
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    indexes = shuffle(np.arange(x_test.shape[0]))
    x_test = x_test[indexes]

    print('Total: {} instances, train: {} instances, test: {} instances'.format(               x_data.shape[0], x_train.shape[0], x_test.shape[0]))
    return x_train, x_test


if __name__ == '__main__':
    struct_log = "./data/HDFS/HDFS_100k.log_structured.csv"

    ## 1. 加载日志文件 提取特征向量
    x_train, _ = load_HDFS(struct_log)
    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')

    ## 2. Train an unsupervised model
    print('Train phase:')
    # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    model = PCA() 
    # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    model.fit(x_train)
    # Make predictions and manually check for correctness. Details may need to go into the raw logs
    y_train = model.predict(x_train) 
    print(f"y_train: {y_train}")


    ## 3. Use the trained model for online anomaly detection
    print('Test phase:')
    # Load another new log file. Here we use struct_log for demo only
    x_test, _ = load_HDFS(struct_log)
    # Go through the same feature extraction process with training, using transform() instead
    x_test = feature_extractor.transform(x_test) 
    # Finally make predictions and alter on anomaly cases
    y_test = model.predict(x_test)
    print(f"y_test: {y_test}")

    