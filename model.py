# coding: utf-8
# from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

# 从概率最大的前n个字符中，根据概率分布随机挑选一个字符作为下一个字符
# preds为预测各字符在下一次出现的概率序列

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds) #去除size为1的维度
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符, 1表示有重复值存在
    c = np.random.choice(vocab_size, 1, p=p)[0] #从vocab_size个字符中选择一个字符，概率分布为p
    return c


class CharRNN:
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True: # 用于预测
            num_seqs, num_steps = 1, 1 # 仅仅根据前面一个字符预测后面一个字符
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        # 清除每次运行中的当前图形，以避免变量重复
        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    # 定义输入层
    #
    def build_inputs(self):
        with tf.name_scope('inputs'): # 为了解决命名冲突, 方便地管理参数命名
            # placeholder()函数是在神经网络构建graph的时候在模型中的占位, 创建变量，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
            # 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                # 将inputs转化为one-hot类型数据输出(维度为num_seqs*num_steps*num_classes)
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    # get_variable(name, shpae) 创建变量的时候必须要提供name
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    # 找到embedding data中的对应行下的vector, 其中对应行有inputs给出
                    # 选取一个张量里面索引对应的元素。比如inputs = [1, 3, 5]，则找出embeddings中第1，3，5行，组成一个tensor返回。
                    # 本例中inputs为一个二维数组，则返回的便是一个三维张量。
                    # lstm_inputs维度为num_seqs*num_steps*embedding_size，每个单词得到一个向量
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    # 定义深层神经网络
    def build_lstm(self):
        # 创建单个cell
        def get_a_cell(lstm_size, keep_prob):
            # #定义一个基本的LSTM结构作为循环体的基础结构
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            # 对每个state的节点数量进行dropout
            # 该类通过两个参数控制dropout的概率，一个参数为input_keep_prob，它用来控制输入的dropout概率。
            # 另一个为output_keep_prob，它可以用来控制输出的dropout概率
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        # 堆叠多层cell
        with tf.name_scope('lstm'):
            # 通过MultiRNNCell类实现深层循环神经网络中每一时刻的前向传播过程
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            # cell状态初始化，h_0
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            # 对于单个 RNNCell ，使用call 函数进行运算时，只在序列时间上前进了一步 ，如使用 x1、 ho 得到此h1，通过 x2 、h1 得到 h2 等
            # 如果序列长度为n，要调用n次call函数. 对此提供了一个tf.nn.dynamic_mn函数，使用该函数相当于调用了n次call函数。
            # 通过{ho, x1 , x2，…，xn} 直接得到{h1 , h2，…，hn} 。
            # outputs和state两个输出, output是每个cell输出的叠加; state是一个元组类型的数据,有(c和h两个变量)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1) # 拼接tensor张量, 第二个参数是指定维度
            # 将每个batch的每个state拼接成一个二维的batch_size * state_node_size(lstm_size)列矩阵
            x = tf.reshape(seq_output, [-1, self.lstm_size]) # 将每个batch的每个state拼接成batch_size*lstm_size

            # 构建输出层softmax
            with tf.variable_scope('softmax'):
                # 初始化输出的权重， 共享
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            # 归一化
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    # 定义损失函数
    def build_loss(self):
        with tf.name_scope('loss'):
            # 对输出进行one-hot编码
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            # 交叉熵计算
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            # 计算平均损失函数
            self.loss = tf.reduce_mean(loss)

    # 定义梯度裁剪, 防止梯度爆炸
    def build_optimizer(self):
        # 使用clipping gradients, 避免梯度计算迭代过程变化过大导致梯度爆炸现象
        tvars = tf.trainable_variables()
        # gradients为计算梯度(loss对tvars参数求偏导)，梯度修剪
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        # 在优化器中应用梯度修剪，zip将多个序列合并成元组
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # 训练网络
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # 控制台输出控制
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    # 用一个字符生成一段文本
    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))