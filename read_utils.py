import numpy as np
import copy #引入该模块的浅拷贝，不改变源对象的值
import time
import tensorflow as tf
import pickle #该模块能序列化对象并保存在磁盘中
import logging
import collections as col # 集合类型collections提供了defaultdict()方法

## 该模块为数据预处理的实现模块。主要是一个文本转换类和batch生成器，文本转换类是对字符按频数排序，取频数高的若干字符，并且得到相应的index数组。

# batch生成函数：batch生成器根据得到的index数组，生成batch
def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1)) # 列数未知, 转变为n_seqs行
    while True:
        np.random.shuffle(arr) #将二维数组的第二维随机打乱
        for n in range(0, arr.shape[1], n_steps): # range(start,end,steps)
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1] = x[:, 1:]
            y[:, -1] = x[:, 0]
            yield x, y


#文本转换类，将word与id进行转换
class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            # 如果已经存在字典文件, 即(将字符集合进行编码的字典文件)
            # Pickle模块实现了基本的数据序列与反序列化。
            # pickle.load() 反序列化对象，将文件中的数据解析为一个python对象
            with open(filename, 'rb') as f:  # rb 以二进制格式打开一个文件用于只读
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)  # #组成text的所有字符，重复的被删除
            # logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等
            logging.info('组成文本的字符集合:')
            logging.info("数量： %d" % len(vocab))
            # defaultdict(factory_function) 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
            # factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，
            # 比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
            vocab_count = col.defaultdict(int) # 将单词表初始化为一个字典
            # 统计字符的频数
            for word in text:
                vocab_count[word] += 1
            # vocab_cout是字典类型, 为二维数据, 第一列是字符名称, 第二列是频数, 将其转换成list数据
            vocab_count_list = list(vocab_count.items())
            # sort(x, key=lambda x:x[1], reverse=True) 将对象x的第二列数据(频数)进行降序排列
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)  # 根据频数降序排序
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab] # 截取从0开始到允许的最大长度
            vocab = [x[0] for x in vocab_count_list] # 获取前max_vocab的字符名
            self.vocab = vocab
        # 对vocab进行编序
        # 得到字符:字符index,enumerate将可循环序列sequence(self.vocab)以start开始分别列出字符index和字符
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        # 得到字典，[字符index:字符]
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self): # 将vocab_size作为一个变量成员调用而不是方法
        return len(self.vocab) + 1 # # 加上一个未登录词

    def word_to_int(self, word): # 根据给定的字符返回index
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab) # 未登录词---最后一个序号

    def int_to_word(self, index): # 根据给定indx返回字符
        if index == len(self.vocab):
            return '<unk>' # 未登录词
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text): # 将文本序列化：字符转化为index
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr): # 反序列化
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    # 存储词典
    def save_to_file(self, filename):
        # pickle.dump(obj, file, protocol=None) 序列化对象，将对象obj保存到文件file中去
        # obj表示将要封装的对象; file表示obj要写入的文件对象; file必须以二进制可写模式打开，即“wb”
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)