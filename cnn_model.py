
# coding: utf-8

import tensorflow as tf

class CNNConfig(object):
    """CNN参数"""
    embedding_dim = 64 #词向量维度
    seq_length = 600 #处理序列长度
    num_class = 7 #类别数
    num_filters = 256 #卷积核数目
    kernel_size = 5 #卷积核尺寸为5，未采用论文2，3，4的处理
    vocab_size = 5000 #词汇表大小

    hidden_dim = 128 #全连接层神经元

    dropout_keep_prop = 0.5 #dropout比例
    learning_rate = 1e-3 #学习率
    batch_size = 64 #每批训练大小
    num_epochs = 10 #迭代轮次

    print_per_batch = 100 #每多少批输出一次结果
    save_per_batch = 10 #每多少批存入tensorboard


class TextCNN(object):
    """CNN文本分类模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_class], name='input_y')
        self.keep_prop = tf.placeholder(tf.float32, name='keep_prop')

        self.cnn()

    def cnn(self):
        '''cnn模型'''

        #词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            # 完成此操作后 embedding_inputs的形状为[input_x, embedding_dim]，训练结束后，embedding将包含词汇表中所有词的词向量
            # 详见 https://www.tensorflow.org/guide/embedding?hl=zh-cn

        with tf.name_scope("cnn"):
            # CNN layer,一维卷积，输入为600 * 词向量维度，卷积核个数num_filters，卷积核尺寸kernel_size，一维卷积只需要kernel_size就行,即不需要n×n，只要n即可
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling 最大池化
            gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")

        with tf.name_scope("full_connection"):
            # 全连接层，后接dropout和relu
            # 添加一个全连接，输入为gmp，全连接神经元个数为hidden_dim
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            fc = tf.layers.dropout(fc, self.config.dropout_keep_prop)
            fc = tf.nn.relu(fc)

            # 输出层
            self.logits = tf.layers.dense(fc, self.config.num_class, name="fc2")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1) #预测分类

        with tf.name_scope("optimize"):
            # 优化器,
            # 损失函数为交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) # 计算损失
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls) #tf.equal用于逐个判断两向量对应元素是否相等
            # tf.cast将correct_pred（true，false形式）转为float32形式，然后计算平均值，也就是true转为1，然后计算所有1的个数，除以类别总数即为正确率
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

