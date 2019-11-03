# -*- coding: utf-8 -*-
# @Time     :2019/11/3 9:58
# @Author   :XiaoMa
# @File     :CBOW.py

import collections
import math
import numpy as np

import random
import tensorflow as tf

from matplotlib import pylab
from six.moves import range

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import csv

## 更改数据生成过程
#我们需要为CBOW定义一个新的数据生成器.新输入数组的shape是（batch_size，context_window * 2）.
# 也就是说，CBOW中的一个批处理捕获给定单词的上下文中的所有单词.

def read_data():
    """
    提取包含在zip文件中的第一个文件作为单词列表，并使用nltk库对其进行预处理。
    """
    with open('/E/home/mayajun/PycharmProjects/word2vec/skip-gram/text8.txt')  as file:
        files=file.read()
        files=files.lower()
        str_list=files.split(' ')

    return str_list

words = read_data()
print('数据大小 %d' % len(words))
print('示例单词（开始）: ', words[:10])
print('示例单词（结束）: ', words[-10:])

## 创建字典（Dictionaries）
# 构建以下内容.为了理解这些元素，让我们假设文本内容为“I like to go to school”
# *`dictionary`: 将字符串单词映射到ID(例如 {I: 0, like: 1, to: 2, go: 3, school: 4}) *`reverse_dictionary`: 将ID映射到字符串单词(例如
# {0: I, 1: like, 2: to, 3: go, 4: school} * `count`: （单词，频率）元素列表(例如[(I, 1), (like, 1), (to, 2), (go, 1), (school, 1)]
# * `data`: 包含我们读取的文本字符串，其中字符串单词被替换为单词ID(例如[0, 1, 2, 3, 2, 4])

# 它还引入了一个额外的特殊标记 `UNK`，表示稀有单词太少，无法使用.
# 我们将词汇量大小限制为50000
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # 仅获取vocabulary_size最常用的单词作为词汇表
    # 所有其他单词将替换为UNK令牌（标记）
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    # 通过给出字典的当前长度为每个单词创建一个ID
    # 并将该项添加到字典中
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    # 遍历我们拥有的所有文本并生成一个列表，其中每个元素对应于在该索引处找到的单词的ID
    for word in words:
        # 如果单词在词典中则使用单词ID，否则使用特殊标记“UNK”的ID
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # 字典（dictionary）['UNK']
            unk_count = unk_count + 1
        data.append(index)

    # 使用UNK出现次数来更新COUNT变量统计次数
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # 确保字典的大小与词汇量大小相同
    assert len(dictionary) == vocabulary_size
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('常见的词 (+UNK)', count[:5])
print('样本数据', data[:10])
del words  # 减少内存

data_index = 0


def generate_batch_cbow(batch_size, window_size):
    # window_size是我们从给中心单词（目标单词）的每一侧看到的单词数量
    # 创建一个batch
    # 每当我们读取一组数据点时，data_index就会更新1
    global data_index

    # span定义了总窗口大小，我们在实例中考虑的数据如下所示.
    # [ skip_window target skip_window ]
    # 例如 如果 skip_window = 2 ，那么 span = 5
    span = 2 * window_size + 1  # [ skip_window target skip_window ]

    # 两个numpy数组用于保存目标词（批处理）和上下文词（标签）
    # 注意，批处理具有span-1 = 2 * window_size的列
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # 缓冲区保存span中包含的数据
    buffer = collections.deque(maxlen=span)

    # 填充缓冲区并更新data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 这里，进行批量读取
    # 我们遍历每个批处理的索引
    # 对于每个批处理索引，我们遍历span的元素以填充批处理数组的列
    for i in range(batch_size):
        target = window_size  # 目标标签位于缓冲区的中心
        target_to_avoid = [window_size]  # 我们只需要知道给定单词周围的单词，而不是单词本身

        # 将所选目标词添加到avoid_list以供下次使用
        col_idx = 0
        for j in range(span):
            # 创建批处理时忽略目标词
            if j == span // 2:
                continue
            batch[i, col_idx] = buffer[j]
            col_idx += 1
        labels[i, 0] = buffer[target]

        # 每次读取数据点时，我们都需要将span移动1以创建新的span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


for window_size in [1, 2]:
    data_index = 0
    batch, labels = generate_batch_cbow(batch_size=8, window_size=window_size)
    print('\n使用 window_size = %d:' % (window_size))
    print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

### 定义超参数

# 这里我们定义几个超参数，包括
# `batch_size`（单个批次中的样本量）`embedding_size`（嵌入向量的大小）`window_size`（上下文窗口大小）．

batch_size = 128  # 一个batch中的数据点数
embedding_size = 128  # 嵌入向量的维数
# 中心词（目标词）左右侧要考虑单词的个数.
# 根据设计，Skip-gram不需要在给定步长中具有所有上下文单词
# 但是，对于CBOW而言，这是有要求的，因此我们限制窗口大小
window_size = 2

# 我们选择一个随机验证集来对最近邻居进行采样
valid_size = 16  # 用于评估单词之间相似性的随机单词集
# 从一个大窗口中随机采样有效数据点而不总是采样确定性的数据点
valid_window = 50

# 在选择有效样例时，我们会选择一些最常用的单词以及一些很少见的单词
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)

num_sampled = 32  # 要采样的负样例数

### 定义输入和输出

#在这里，我们定义了用于输入和输出训练的占位符（每个大小为
#`#batch_size`），以及一个包含验证示例的常数张量.

tf.reset_default_graph()

# 训练输入数据（目标单词ID），注意，它有2 * window_size大小的列
train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
# 训练输入标签数据（上下文单词ID）
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# 验证输入数据，我们不需要占位符
# 因为我们已经定义了选择作为用于评估单词向量的验证数据的单词ID
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# %% md

### 定义模型参数和其他变量
#我们现在定义了几个TensorFlow变量，例如嵌入层（`embeddings`）和神经网络参数（`softmax_weights`和
#`softmax_biases`）

# 变量
# 嵌入图层，包含单词嵌入
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))

# Softmax 权重和偏差
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=0.5 / math.sqrt(embedding_size), dtype=tf.float32))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))


### 定义模型计算

# 我们首先定义查找函数以获取一组给定输入的相应的嵌入向量. 具体来说，我们定义2 倍的 `window_size` 嵌入查找.然后我们连接所有这些查找到的嵌入向量以组成大小为
# `[batch_size，embedding_size，2 * window_size]`的矩阵．之后，我们对这些嵌入向量进行平均化处理，以产生大小为
# `[batch_size，embedding_size]`的平均嵌入．在此基础上，我们就可以定义负采样损失函数 `tf.nn.sampled_softmax_loss`，它接受嵌入向量和先前已定义的神经网络参数．

# 模型
# 查找一批输入的嵌入
# 这里,我们为输入占位符中的每一列进行嵌入查找,然后对它们求平均以生成embedding_size词向量
stacked_embedings = None
print('定义 %d 个嵌入查找,表示上下文中的每个单词' % (2 * window_size))
for i in range(2 * window_size):
    embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
    x_size, y_size = embedding_i.get_shape().as_list()
    if stacked_embedings is None:
        stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
    else:
        stacked_embedings = tf.concat(axis=2, values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])

assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
print("叠加嵌入大小: %s" % stacked_embedings.get_shape().as_list())
mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)
print("减少嵌入的平均大小: %s" % mean_embeddings.get_shape().as_list())

# 每次使用负标签样本计算softmax损失
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                               labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

### 模型参数优化器

#然后我们定义一个恒定的学习率和一个使用Adagrad方法的优化器.也可以随意尝试列出的其他优化器
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

### 计算单词相似度

#我们根据余弦距离计算两个给定单词之间的相似性.为了有效地执行此操作，我们使用矩阵运算来执行此操作，如下所示.

# 计算小批量（minibatch）示例和所有嵌入之间的相似性.
# 我们使用余弦距离（cosine distance）函数:
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


## 运行 CBOW 算法
num_steps = 100001
cbow_losses = []

# ConfigProto是一种提供执行图所需的各种配置的方法
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # 初始化图中的变量
    tf.global_variables_initializer().run()
    print('初始化')

    average_loss = 0

    # 训练Word2vec模型进行num_step迭代
    for step in range(num_steps):
        # 生成一批数据
        batch_data, batch_labels = generate_batch_cbow(batch_size, window_size)

        # 填充feed_dict，运行优化器（最小化损失）并计算损失
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        # 更新平均损失变量
        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
                # 平均损失是对过去2000批次损失的估计
            cbow_losses.append(average_loss)
            print('第 %d 步长上的平均损失: %f' % (step + 1, average_loss))
            average_loss = 0

        # 评估验证集单词的相似性
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            # 在这里，我们根据余弦距离为给定验证单词计算top_k最接近的单词
            # 我们对验证集中的所有单词执行此操作
            # 注意：这一步的计算成本很高

            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # 最近邻数量
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = '与 %s 最接近的单词:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    cbow_final_embeddings = normalized_embeddings.eval()
