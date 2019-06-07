# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import keras as kr
from collections import Counter

from sklearn import metrics
from cnn_model import CNNConfig, TextCNN

import time
from datetime import timedelta



train_dir = "./data/train.txt"
test_dir = "./data/test.txt"
val_dir = "./data/val.txt"
vocab_dir = "./data/vocab.txt"

save_dir = "checkpoints/textcnn"
save_path = "checkpoints/textcnn/best_validation" # 最佳验证结果保存路径



# 耗时
def get_time_use(start_time):
    end_time = time.time()
    time_use = end_time - start_time
    return timedelta(seconds=int(round(time_use)))


def read_category():
    '''读取分类目录'''
    categories = ['财经', '房产', '股票', '家居', '科技', '时政', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories)))) # 标签与0-6的编号打包，变成体育：0这种字典形式
    return categories, cat_to_id

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    print(labels)
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    print(len(data_train))
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, 'w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    '''读取词汇表'''
    with open(vocab_dir, 'r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    words_to_id = dict(zip(words, range(len(words))))
    return words, words_to_id

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '''将文件转为id表示'''
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass

    data_id, label_id = [], []
    for i in range(len(contents)):
        # data_id列表中的元素是每个content的列表
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 用Keras的pad_sequence来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length,padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id)) # 标签转为one_hot

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    '''生成批次数据'''
    data_len = len(x)
    num_batch = int((data_len-1) / batch_size)+1
    indices = np.random.permutation(np.arange(data_len))
    # permutation用于对原数组洗牌，permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]


def evaluate(sess, x_, y_):
    '''评估在某一数据集上的准确率和损失'''
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0
    total_acc = 0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prop: 1.0}
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss*batch_len
        total_acc += acc*batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring tensorboard and saver")
    tensorboard_dir = "tensorboard/textcnn"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data")
    # 载入训练集验证集
    start_time = time.time()
    # 训练集和验证集
    x_trian, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_use = get_time_use(start_time)
    print("Time usage", time_use)

    # 创建Session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("training and evaluating...")
    start_time = time.time()
    total_batch = 0 # 总批次
    best_acc_val = 0 # 最佳验证集准确率
    last_improved = 0 # 记录上一次提升批次
    require_improvement = 1000 # 超过1000次未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch:", epoch+1)
        batch_train = batch_iter(x_trian, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prop: config.dropout_keep_prop}

            if total_batch % config.save_per_batch == 0:
                # 多少轮次将结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次打印在训练集和验证集上的性能
                feed_dict[model.keep_prop] = 1.0 # 打印性能时停止使用dropout
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)

                if acc_val > best_acc_val: # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_use = get_time_use(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_use, improved_str))

            session.run(model.optim, feed_dict=feed_dict) # 运行优化
            total_batch += 1
            if total_batch - last_improved > require_improvement:
                # 验证集正确率不提升，提前结束训练
                print("No optimizer for a long time, auto-stopping")
                flag = True
                break # 退出循环
        if flag:
            break



def test():
    print("Loading test data")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path) # 读取保存的模型

    print("testing")
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size)+1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch): # 逐批次处理
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        feed_dict = {model.input_x: x_test[start_id:end_id], model.keep_prop: 1.0}
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall, F1-score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    #混淆矩阵
    print("Confusion matrix")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_use = get_time_use(start_time)
    print("time_use:", time_use)




if __name__ == "__main__":

    print("Configuring CNN model")
    config = CNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category() # 类别与ID
    words, word_to_id = read_vocab(vocab_dir) # 词与词id
    config.vocab_size = len(words)
    model = TextCNN(config)

    train()

    test()

    # process_file(train_dir, word_to_id, cat_to_id)

