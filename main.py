import argparse
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

FLAGS = None


def load_data():
    """
    Load data from pickle
    :return: Arrays
    """
    with open(FLAGS.source_data, 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        return data_x, data_y, word2id, id2word, tag2id, id2tag


def get_data(data_x, data_y):
    """
    Split data from loaded data
    :param data_x:
    :param data_y:
    :return: Arrays
    """
    print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    print('Data X Example', data_x[0])
    print('Data Y Example', data_y[0])
    
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40)
    
    print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    print('Valid X Shape', valid_x.shape, 'Valid Y Shape', valid_x.shape)
    print('Test Y Shape', test_x.shape, 'Test Y Shape', test_y.shape)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_generator(data_x, data_y, batch_size):
    """
    Get generator of x, y data
    :param data_x:
    :param data_y:
    :return: Iterator
    """
    dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def main():
    # Load data
    data_x, data_y, word2id, id2word, tag2id, id2tag = load_data()
    # Split data
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data(data_x, data_y)
    
    vocab_size = len(word2id) + 1
    print('Vocab Size', vocab_size)
    
    # Input Layer
    with tf.variable_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.time_step], name='x_input')
        y_label = tf.placeholder(tf.int32, [None, FLAGS.time_step], name='y_label')
    
    # Embedding Layer
    with tf.variable_scope('embedding'):
        embedding = tf.Variable(tf.random_normal([vocab_size, FLAGS.embedding_size]), dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, x)
    
    # Variables
    batch_size = tf.placeholder(tf.int32, [])
    keep_prob = tf.placeholder(tf.float32, [])
    
    # RNN Layer
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
    initial_state_fw = cell_fw.zero_state(x.shape[0], tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
    print('Inputs', inputs)
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
    print('Inputs unstack', inputs)
    output, _, _ = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, initial_state_fw=initial_state_fw,
                                                  initial_state_bw=initial_state_bw)
    # output_fw, _ = tf.nn.dynamic_rnn(cell_fw, inputs=inputs, initial_state=initial_state_fw)
    # output_bw, _ = tf.nn.dynamic_rnn(cell_bw, inputs=inputs, initial_state=initial_state_bw)
    # print('Output Fw, Bw', output_fw, output_bw)
    # output_bw = tf.reverse(output_bw, axis=[1])
    # output = tf.concat([output_fw, output_bw], axis=2)
    output = tf.stack(output, axis=1)
    print('Output', output)
    output = tf.reshape(output, [-1, FLAGS.num_units * 2])
    print('Output Reshape', output)
    
    # Output Layer
    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b
    print('Output Y', y)
    
    y_label_reshape = tf.reshape(y_label, [-1])
    print('Y Label Reshape', y_label_reshape)
    
    # Prediction
    correct_prediction = tf.equal(tf.cast(tf.argmax(y, axis=1), tf.int32), y_label_reshape)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Prediction', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
                                                                                  logits=tf.cast(y, tf.float32)))
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    
    # Iterator
    train_iterator = get_generator(train_x, train_y, FLAGS.batch_size)
    sess = tf.Session()
    sess.run(train_iterator.initializer)
    sess.run(tf.global_variables_initializer())
    
    # Saver
    saver = tf.train.Saver()
    
    # Train
    for step in range(0, FLAGS.steps + 1):
        batch_x, batch_y = sess.run(train_iterator.get_next())
        sess.run(train,
                 feed_dict={x: batch_x, y_label: batch_y, batch_size: FLAGS.batch_size, keep_prob: FLAGS.keep_prob})
        
        loss, acc = sess.run([cross_entropy, accuracy],
                             feed_dict={x: batch_x, y_label: batch_y, batch_size: FLAGS.batch_size,
                                        keep_prob: FLAGS.keep_prob})
        print('Train Loss ', loss, 'Accuracy', acc)
        
        print(test_x.shape, test_y.shape, valid_x.shape, valid_y.shape)
        if step % FLAGS.steps_per_valid == 0:
            print('Valid Accuracy',
                  sess.run(accuracy, feed_dict={x: valid_x, y_label: valid_y, batch_size: valid_x.shape[0],
                                                keep_prob: 1}))
        if step % FLAGS.steps_per_test == 0:
            print('Test Accuracy',
                  sess.run(accuracy, feed_dict={x: test_x, y_label: test_y, batch_size: test_x.shape[0],
                                                keep_prob: 1}))
        # if step % FLAGS.steps_per_save == 0:
        #     save_path = 'ckpt/bi-lstm-{}'.format()
        #     saver.restore(sess=sess, save_path='d')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--batch_size', help='batch size', default=50)
    parser.add_argument('--source_data', help='source size', default='./data/data.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)
    parser.add_argument('--num_units', help='num of units', default=64, type=int)
    parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--embedding_size', help='time steps', default=64, type=int)
    parser.add_argument('--category_num', help='category num', default=5, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--epoch_num', help='num of epoch', default=1000, type=int)
    parser.add_argument('--steps_per_test', help='steps per test', default=1000, type=int)
    parser.add_argument('--steps_per_valid', help='steps per valid', default=100, type=int)
    parser.add_argument('--steps_per_save', help='steps per save', default=2000, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.5, type=float)
    
    FLAGS = parser.parse_args()
    print(FLAGS)
    main()
