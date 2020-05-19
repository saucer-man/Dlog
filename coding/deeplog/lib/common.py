import pickle
from datetime import datetime

# 将对象以二进制形式保存
def save(filename, cls):
        with open(filename, 'wb') as f:
            pickle.dump(cls, f)


# 加载二进制形式的对象
def load(filename):
    with open(filename, 'rb') as f:
        cls = pickle.load(f)
        return cls


def time_elapsed(time_front, time_back, format="%H:%M:%S"):
    try:
        time_front_array = datetime.strptime(time_front, format)
        # print(time_front_array)
        time_back_array = datetime.strptime(time_back, format)
        #print(time_back_array)
        # time_elapsed = time_back_stamp - time_front_stamp
        time_elapsed = (time_back_array - time_front_array).seconds
        # if time_elapsed > 100:
        #     print(f"{time_front_array}  {time_back_array}  {time_elapsed}")
        return str(time_elapsed)
    except:
        return "0"

