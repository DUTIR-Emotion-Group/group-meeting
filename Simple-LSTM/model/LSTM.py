import tensorflow as tf


class LSTM:
    def __init__(self, embbedding_matrix, train_x, train_y, test_x, test_y):

        self.embbedding_matrix = embbedding_matrix
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.learning_rate = 0.01
        self.batch_size = 128
        self.max_sentence_length = 256
        self.class_num = 2
        self.epoch = 100
        self.hidden_layer_num = 50

        self.sess = tf.Session()

    def build_model(self):
        with tf.name_scope("input"):
            self.input_sentence = tf.placeholder(tf.int32, shape=[None, self.max_sentence_length])
            self.input_label = tf.placeholder(tf.int32, shape=[None, self.class_num])

        with tf.name_scope("embedding_layer"):
            input_sentence_embedding = tf.nn.embedding_lookup(self.embbedding_matrix, self.input_sentence)

        with tf.name_scope("lstm_layer"):
            hidden_output, cell_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.hidden_layer_num),
                inputs=input_sentence_embedding,
                time_major=False,
                dtype=tf.float32,
                scope="lstm_layer"
            )
            last_hidden_output = cell_state[1]

        with tf.name_scope("softmax_layer"):
            W_s = tf.get_variable(name='W_s',shape=[self.hidden_layer_num, self.class_num],initializer=tf.random_uniform_initializer(-0.1, 0.1), dtype=tf.float32)
            b_s = tf.get_variable(name='b_s', shape=[1, self.class_num], initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.final_scores = tf.tanh(tf.matmul(last_hidden_output, W_s) + b_s)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_label,logits=self.final_scores)
            self.loss = tf.reduce_mean(self.cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        with tf.name_scope("predict"):
            self.correct_pred = tf.cast(tf.equal(tf.argmax(self.final_scores, 1), tf.argmax(self.input_label, 1)),tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_pred)

    def train(self):
        self.train_pointer = 0
        self.max_train_length = int(self.train_x.shape[0])
        total_loss_sum = 0
        flag, feed_dict = self.get_train_batch()
        while flag:
            _, batch_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            total_loss_sum += int(feed_dict[self.input_sentence].shape[0]) * float(batch_loss)
            flag, feed_dict = self.get_train_batch()
        return total_loss_sum/self.max_train_length

    def test(self):
        self.test_pointer = 0
        self.max_test_length = int(self.test_x.shape[0])
        flag, feed_dict = self.get_test_batch()
        total_loss_sum = 0
        total_corrected_count = 0
        while flag:
            batch_accuracy, batch_loss = self.sess.run([self.accuracy, self.loss], feed_dict=feed_dict)
            total_loss_sum += int(feed_dict[self.input_sentence].shape[0]) * float(batch_loss)
            total_corrected_count += int(feed_dict[self.input_sentence].shape[0]) * float(batch_accuracy)
            flag, feed_dict = self.get_test_batch()
        return total_loss_sum / self.max_test_length,total_corrected_count/self.max_test_length
        #
        # feed_dict = self.get_test_data()
        # loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        # return loss, accuracy

    def get_train_batch(self):
        if self.train_pointer < self.max_train_length:
            begin_index = self.train_pointer
            self.train_pointer = self.train_pointer + self.batch_size
            end_index = min(self.train_pointer, self.max_train_length)
            sample = {
                self.input_sentence: self.train_x[begin_index:end_index],
                self.input_label: self.train_y[begin_index:end_index]
            }
            return True, sample
        else:
            self.train_pointer = 0
            return False, None

    def get_test_batch(self):
        if self.test_pointer < self.max_test_length:
            begin_index = self.test_pointer
            self.test_pointer = self.test_pointer + self.batch_size
            end_index = min(self.test_pointer, self.max_test_length)
            sample = {
                self.input_sentence: self.test_x[begin_index:end_index],
                self.input_label: self.test_y[begin_index:end_index]
            }
            return True, sample
        else:
            self.test_pointer = 0
            return False, None


    def run(self):
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        print("模型创建完成！")
        print("模型共训练{}轮,每{}轮输出一次结果".format(self.epoch,5))
        for epoch in range(self.epoch):
            train_loss = self.train()
            if epoch % 5 == 0:
                print("current epoch:{},train_loss:{}".format(epoch, train_loss))
                test_loss, accuracy = self.test()
                print("test_loss:{},test_accuracy:{}".format(test_loss, accuracy))

