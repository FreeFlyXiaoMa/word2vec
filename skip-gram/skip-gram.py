# -*- coding: utf-8 -*-
# @Time     :2019/10/29 9:42
# @Author   :XiaoMa
# @File     :skip-gram.py

#  Skip-gram 和 CBOW模型
# ** 注意 **: 此代码可能会占用大量内存。 如果你有大量的RAM内存（ > 4 GB），你不必担心。 否则请减少 `batch_size`
# 或 `embedding_size` 参数以允许模型获得适合内存。

# 这些都是我们稍后将要使用的模块.在继续操作之前,请确保可以导入它们

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

## 数据集（Dataset）
# 此代码下载[dataset]（http: // www.evanjones.ca / software / wikipedia2text.html）
# ，其中包含多篇维基百科文章，总计大约61兆字节.此外，代码确保文件在下载后具有正确的大小.

url = 'http://mattmahoney.net/dc/'

## 使用NLTK进行预处理读取数据
# 将原始数据读取到字符串，转换为小写并使用nltk库对其进行标记.此代码以1MB大小的份数读取数据，因为一次处理全文会减慢任务速度，最后返回单词列表.
# 必要的标记器需要我们下载.

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
#
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

## 为Skip-Gram生成批量数据 生成批处理或目标词(`批处理`) 和一批相应的上下文词(`标签`)． 它一次读取总窗口大小为
# `2 * window_size + 1` 的单词(称为 `span`) ， 并在单个范围内创建 `2 * window_size` 数据点。
# 该函数以这种方式继续，直到创建 `batch_size` 数据点.每当我们到达单词序列的末尾时，我们就从头开始.

data_index = 0
def generate_batch_skip_gram(batch_size, window_size):
    # 每次读取数据点时，data_index都会更新1
    global data_index
    # 两个numpy数组来保存目标词（批处理）和上下文词（标签）
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # span定义了总窗口大小，我们在实例中考虑的数据如下所示
    # [skip_window target skip_window]
    span = 2 * window_size + 1

    # 缓冲区保存span中包含的数据
    buffer = collections.deque(maxlen=span)

    # 填充缓冲区并更新data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 这是我们为单个目标单词采样的上下文单词的数量
    num_samples = 2 * window_size

    # 我们将批量读取分成两个for循环来进行
    # 内部for循环使用包含span的数据填充带有num_samples数据点的批处理和标签
    # 循环输出器对batch_size//num_samples 重复此操作以生成完整批处理

    for i in range(batch_size // num_samples):
        k = 0
        # 避免目标词本身作为预测
        # 填充批处理和标签numpy数组
        for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            k += 1

            # 每当我们读取num_samples数据点时，我们已经创建了单个跨度(span)可能的最大数据点数，
        # 因此我们需要将跨度(span)移动1以创建新的跨度(span)
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])

for window_size in [1, 2]:
    data_index = 0
    batch, labels = generate_batch_skip_gram(batch_size=8, window_size=window_size)
    print('\n使用 window_size = %d:' % window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


## Skip-Gram 算法

### 定义超参数
# 这里我们定义几个超参数，包括
# `batch_size`（单个批次中的样本量）`embedding_size`（嵌入向量的大小）`window_size`（上下文窗口大小）.


batch_size = 128
embedding_size = 128  # 词向量的维数
window_size = 4  # 左右两边各考虑多少个词

# 我们选择一个随机验证集来对最近邻进行采样
valid_size = 16  # 用于评估相似性的随机词集
# 仅在分布的头部选择开发样本
valid_window = 50

# 在选择有效样例时，我们会选择一些常用词以及一些很少见的单词
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)

num_sampled = 32  # 要抽样的负样例数量

### 定义输入和输出
# 在这里，我们定义了用于输入和输出训练的占位符（每个大小为 `batch_size`），以及一个包含验证示例的常量张量．


tf.reset_default_graph()

# 训练输入数据（目标单词ID）
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
# 训练输入标签数据（上下文单词ID）
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# 验证输入数据，我们不需要占位符
# 因为我们已经定义了选择作为用于评估单词向量的验证数据的单词ID
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

### 定义模型参数和其他变量
# 我们现在定义了几个TensorFlow变量，例如嵌入层（`embeddings`）和神经网络参数（`softmax_weights` 和 `softmax_biases`）

# 变量
# 嵌入图层，包含单词嵌入
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Softmax权重和偏差
softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=0.5 / math.sqrt(embedding_size))
)
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))


### 定义模型计算
# 我们首先定义查找函数以获取一组给定输入的相应的嵌入向量.有了它，我们定义负采样损失函数
# `tf.nn.sampled_softmax_loss`，它接受嵌入向量和先前定义的神经网络参数.

# 模型
# 查找一批输入的嵌入
embed = tf.nn.embedding_lookup(embeddings, train_dataset)

# 每次使用负标签样本计算softmax损失.
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_weights, biases=softmax_biases, inputs=embed,
        labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size)
)

### 计算单词相似度
# 我们根据余弦距离计算两个给定单词之间的相似度.为了有效地执行此操作，我们使用矩阵运算来执行此操作，如下所示.

### 模型参数优化器

# 然后我们定义一个恒定的学习率和一个使用Adagrad方法的优化器.也可以随意尝试列出的其他优化器[这里]
# (https: // www.tensorflow.org / api_guides / python / train).

# %%

# 优化器.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
normalized_embeddings=embeddings/norm
valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)
# %% md

## 运行Skip-Gram算法

# %%

num_steps = 100001
skip_losses = []
# ConfigProto是一种提供执行图所需的各种配置设置的方法
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # 初始化图中的变量
    tf.global_variables_initializer().run()
    print('初始化')
    average_loss = 0

    # 进行num_step迭代,训练Word2vec模型
    for step in range(num_steps):

        # 生成一批数据
        batch_data, batch_labels = generate_batch_skip_gram(
            batch_size, window_size)

        # 填充feed_dict，运行优化器（最小化损失）并计算损失
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        # 更新平均损失变量
        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            skip_losses.append(average_loss)
            # 平均损失是对过去2000批次损失的估计
            print('第 %d 步长上的平均损失: %f' % (step + 1, average_loss))
            average_loss = 0

        # 评估验证集单词的相似度
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            # 在这里，我们根据余弦距离为给定验证单词计算top_k最接近的单词
            # 我们对验证集中的所有单词执行此操作
            # 注意：这是一步的计算成本很高
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # 最近邻数量
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = '与 %s 最接近的单词:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    skip_gram_final_embeddings = normalized_embeddings.eval()

# 我们将保存所学的向量词和随着时间推移所造成的损失，因为这些信息后面需要进行相关比较.
np.save('skip_embeddings', skip_gram_final_embeddings)

with open('skip_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_losses)


# %% md

## 可视化Skip-Gram算法的学习

### 找到只聚集在一起的单词而不是稀疏分布的单词

# %%

def find_clustered_embeddings(embeddings, distance_threshold, sample_threshold):
    '''
    仅查找密集聚类的嵌入.
    这样就消除了更稀疏的单词嵌入，使可视化更清晰．
    这对于t-SNE可视化非常有用

    distance_threshold: 相邻点之间的最大距离
    sample_threshold: 需要被视为聚类（群集）的邻居数量
    '''

    # 计算余弦（cosine）相似度
    cosine_sim = np.dot(embeddings, np.transpose(embeddings))
    norm = np.dot(np.sum(embeddings ** 2, axis=1).reshape(-1, 1),
                  np.sum(np.transpose(embeddings) ** 2, axis=0).reshape(1, -1))
    assert cosine_sim.shape == norm.shape
    cosine_sim /= norm

    # 使所有对角线条目为零，否则将被选为最高项
    np.fill_diagonal(cosine_sim, -1.0)

    argmax_cos_sim = np.argmax(cosine_sim, axis=1)
    mod_cos_sim = cosine_sim
    # 如果有超过阈值的n个项目，则查找循环中的最大值以进行计数
    for _ in range(sample_threshold - 1):
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]), argmax_cos_sim] = -1

    max_cosine_sim = np.max(mod_cos_sim, axis=1)

    return np.where(max_cosine_sim > distance_threshold)[0]


# %% md

### 使用Scikit-Learn计算单词嵌入的t-SNE可视化

# 我们将使用大样本空间来构建T-SNE流形(T-SNE manifold)，然后使用余弦相似性对其进行修剪
num_points = 1000

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

print('T-SNE的拟合嵌入.这可能需要一些时间......')
# 获取T-SNE流形
selected_embeddings = skip_gram_final_embeddings[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('裁剪T-SNE嵌入')
# 通过获得仅高于相似性阈值的n个样本的嵌入来修剪嵌入
# 这会使视觉效果更加清晰
selected_ids = find_clustered_embeddings(selected_embeddings, .25, 10)
two_d_embeddings = two_d_embeddings[selected_ids, :]

print('在', num_points, '个样本 ', '之中,', '通过裁剪选择出', selected_ids.shape[0], '个样本')


# %% md

### 用Matplotlib绘制t-SNE结果

# %%

def plot(embeddings, labels):
    n_clusters = 20  # 聚类（簇）数量
    # 自动构建一组离散的颜色，每个颜色都用于聚类
    label_colors = [pylab.cm.nipy_spectral(float(i) / n_clusters) for i in range(n_clusters)]

    assert embeddings.shape[0] >= len(labels), '多于嵌入的标签'

    # 定义 K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
    kmeans_labels = kmeans.labels_

    pylab.figure(figsize=(15, 15))  # in inches

    # 绘制所有嵌入及其相应的单词
    for i, (label, klabel) in enumerate(zip(labels, kmeans_labels)):
        x, y = embeddings[i, :]
        pylab.scatter(x, y, c=label_colors[klabel])

        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom', fontsize=10)

    # 如果需要，用于保存图像
    # pylab.savefig('word_embeddings.png')
    pylab.show()


words = [reverse_dictionary[i] for i in selected_ids]
plot(two_d_embeddings, words)

