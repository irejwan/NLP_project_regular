import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
import tensorflow.contrib.slim as slim
from config import Config

config = Config()


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class RegularRNN:
    def __init__(self, sess):
        state_size = config.RNN.state_size.int
        vocab_size = len(config.Grammar.alphabet.lst)
        self.sess = sess

        # RNN placeholders
        with tf.name_scope('input'):
            self.input_ph = tf.placeholder(tf.int32, [None], name='input_ph')
            input = tf.reshape(tf.one_hot(self.input_ph, depth=vocab_size, name='input_one_hot'),
                               shape=[1, -1, vocab_size])

        self.init_state_ph = tf.placeholder(shape=[None, state_size], dtype=tf.float32, name='init_state_ph')

        stackcell = BasicRNNCell(state_size)
        _, self.final_state = rnn(stackcell, input, initial_state=tf.cast(self.init_state_ph, tf.float32))
        with tf.name_scope('prediction'):
            self.prediction = slim.fully_connected(self.final_state, 1, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.truncated_normal_initializer())

        # RNN loss
        self.label_ph = tf.placeholder(tf.float32, name='label_ph')
        with tf.name_scope('rnn_loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.label_ph))

        with tf.name_scope('rnn_optimizer'):
            global_step = tf.Variable(0, trainable=False)
            # lr = tf.train.exponential_decay(0.001, global_step, 1000, 0.96, staircase=True)
            lr = 1e-3
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, global_step=global_step)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(tf.sigmoid(self.prediction)), self.label_ph), tf.float32))
            variable_summaries(self.accuracy)

    def get_next_state(self, state, input):
        next_state = self.sess.run(self.final_state, feed_dict={self.input_ph: [input],
                                                                self.init_state_ph: [state]})
        return next_state[0]

    def is_accept(self, state):
        y_hat = self.sess.run(self.prediction, feed_dict={self.init_state_ph: state, self.input_ph: []})
        return y_hat[0][0] > 0
