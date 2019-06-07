import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs # 使用codecs模块进行文件操作-读写中英文字符

# 用命令行执行程序时，传递参数
FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_XXX定义一个用于接收 xxx 类型数值的变量, 用于接收从终端输入的命令行
# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch') # 一个 batch 可以组成num_seqs个输入信号序列
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq') # 一个输入信号序列的长度， rnn网络会根据输入进行自动调整
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm') # 隐藏层节点数量，即LSTM的cell中state的数量
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers') # RNN的深度
# 如果中文字符则需要一个word2vec， 字母字符直接采用onehot编码
# word2vec一个NLP工具，它可以将所有的词向量化，这样词与词之间就可以定量的去度量他们之间的关系，挖掘词之间的联系
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding') # 词向量的维度
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
# 不同于英文字符比较短，中文字符比较多，word2vec层之前的输入需要进行one-hot编码，根据字符频数降序排列取前面的3500个编码
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    ## 对数据进行预处理。调用read_utils.py模块中的文本转换类TextConverter，获取经过频数挑选的字符并且得到相应的index。
    ## 然后调用batch_generator函数得到一个batch生成器。
    model_path = os.path.join('model', FLAGS.name) # 路径拼接
    print("模型保存位置: ", model_path)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path) # 递归创建目录
    # Python读取文件中的汉字方法：导入codecs，添加encoding='utf-8'
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        print("建模训练数据来源", FLAGS.input_file)
        text = f.read()
    # 返回一个词典文件
    converter = TextConverter(text, FLAGS.max_vocab)
    # 将经过频数挑选的字符序列化保存
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
    arr = converter.text_to_arr(text) #得到每个字符的index
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps) # 得到一个batch生长期
    print(converter.vocab_size) # 打印字符数量

    ## 数据处理完毕后，调用model.py模块的CharRNN类构造循环神经网络，最后调用train()函数对神经网络进行训练
    model = CharRNN(converter.vocab_size, #字符分类的数量
                    num_seqs=FLAGS.num_seqs, #一个batch中的序列数
                    num_steps=FLAGS.num_steps, #一个序列中的字符数
                    lstm_size=FLAGS.lstm_size, #每个cell的节点数量
                    num_layers=FLAGS.num_layers, #RNN的层数
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


# 使用这种方式保证了，如果此文件被其他文件 import的时候，不会执行main 函数
if __name__ == '__main__':
    tf.app.run()  # 用来处理flag解析，然后执行main函数