import tensorflow as tf
import numpy as np
from data_loader import pre_deal
from keras.preprocessing.sequence import pad_sequences


sequence_length = 100  # 句子长度
num_classes = 2
vacab_size = 10000
embedding_size = 50
filter_size_lst = [2, 3, 4]
num_filters = 50
test_frac = 0.2
batch_size = 128

# load data
df = pre_deal()
df_train = df.iloc[: -1 * int(len(df)*test_frac)]
df_test = df.iloc[-1 * int(len(df)*test_frac):]
train_x = pad_sequences(df_train['id'], sequence_length, padding='post', truncating='post')
train_y = np.array([[1, 0] if i == 0 else [0, 1]
                    for i in df_train['label']])
test_x = pad_sequences(df_test['id'], sequence_length, padding='post', truncating='post')
test_y = np.array([[1, 0] if i == 0 else [0, 1]
                    for i in df_test['label']])
train_size = len(train_x)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# model input
input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

# embedding layer
with tf.name_scope('embedding'):
    embedding_weights = tf.Variable(initial_value=tf.random_uniform([vacab_size, embedding_size], -1.0, 1.0), name='w', trainable=True)
    l_embedding = tf.nn.embedding_lookup(params=embedding_weights, ids=input_x)    # shape=[seq_len, embed_size]
    l_embedding_expend = tf.expand_dims(l_embedding, -1)    # shape=[seq_len, embed_size, 1]. expand channel dimension

# conv and pool layers, including 3 sizes kind of filters
l_pooling_outputs = []
for i, filter_size in enumerate(filter_size_lst):
    with tf.name_scope('conv-%s' % filter_size):
        filter_shape = [filter_size, embedding_size, 1, num_filters]  # in_channels=1, out_channels=num_filters
        w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
        b = tf.Variable(tf.random_uniform([num_filters], -0.1, 0.1), name='bias')
        conv = tf.nn.conv2d(l_embedding_expend, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        l_conv_act = tf.nn.relu(tf.nn.bias_add(conv, b), name='active')

        # max pooling
        l_pooling = tf.nn.max_pool(l_conv_act, ksize=[1, sequence_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                   padding='VALID', name='pooling')
        l_pooling_outputs.append(l_pooling)

# combine feature maps
num_filters_all = num_filters * len(filter_size_lst)
l_concat = tf.concat(l_pooling_outputs, -1)
l_flat = tf.reshape(l_concat, [-1, num_filters_all])

# add dropout
l_dropout = tf.nn.dropout(l_flat, keep_prob=0.2, name='dropout')

with tf.name_scope('output'):
    w = tf.Variable(tf.truncated_normal(shape=[num_filters_all, num_classes]), name='w')
    b = tf.Variable(tf.random_uniform([num_classes], -0.1, 0.1), name='bias')
    scores = tf.nn.xw_plus_b(l_dropout, w, b, name='scores')
    pred = tf.argmax(scores, -1, name='prediction')

with tf.name_scope('loss'):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
    loss_total = tf.reduce_mean(losses)

with tf.name_scope('accuracy'):
    correct_predictions = tf.equal(pred, tf.argmax(input_y, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
train_step = optimizer.minimize(loss=losses, global_step=global_step)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    steps = 1000
    for i in range(steps):
        start_index = (i * batch_size) % train_size
        end_index = min(start_index + train_size, train_size)

        in_x = train_x[start_index: end_index]
        in_y = train_y[start_index: end_index]
        sess.run(train_step, feed_dict={input_x: in_x, input_y: in_y})

        if i % 100 == 0:
            train_loss, train_acc = sess.run([loss_total, accuracy], feed_dict={input_x: train_x, input_y: train_y})
            test_loss, test_acc = sess.run([loss_total, accuracy], feed_dict={input_x: test_x, input_y: test_y})
            print('Step %d, train loos %s, acc %s\n test loss %s, acc %s' % (i, train_loss, train_acc, test_loss, test_acc))








