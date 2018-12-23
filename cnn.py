#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
#忽略warning的输出
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 500
LR = 0.01              # learning rate

def next_batch(data_x, data_y, BATCH_SIZE):
    index = np.random.randint(0, len(data_x), size=BATCH_SIZE)
    b_x = data_x[index]
    b_y = data_y[index]
    #print(b_x.shape)
    return b_x, b_y

data_x = np.loadtxt('data/data_x.txt')
data_y = np.loadtxt('data/data_y.txt')
test_x = data_x[-100:]
test_y = data_y[-100:]
print(data_x.shape)

tf_x = tf.placeholder(tf.float32, [None, 34*32*3])
image = tf.reshape(tf_x, [-1, 34, 32, 3])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 2])            # input y

# CNN
conv1 = tf.layers.conv2d(   # shape (34,32,3)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (34, 32, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (17, 16, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 1, 2)    # -> (7, 7, 32)
print(pool2.shape)
flat = tf.reshape(pool2, [-1, 9*8*32])          # -> (7*7*32, )
output = tf.layers.dense(flat, 2)              # output layer


loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
#tf.argmax可以认为就是np.argmax。tensorflow使用numpy实现的这个API。简单的说，tf.argmax就是返回最大的那个数值所在的下标。

sess = tf.Session()
#tf.global_variables_initializer()返回一个初始化所有全局变量的操作
#tf.local_variables_initializer()返回一个初始化所有局部变量的操作
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)     # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
#
# plt.ion()
for step in range(1000):
    # b_x = data_x
    # b_y = data_y
    b_x, b_y = next_batch(data_x, data_y, BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

#         if HAS_SK:
#             # Visualization of trained flatten layer (T-SNE)
#             tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
#             low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
#             labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')