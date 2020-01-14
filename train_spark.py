
from time import time

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import PhasedLSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn
import logging
from tensorflow.python import debug as tf_debug
import utils
logging.basicConfig(level=logging.INFO)

from utils import *
from config import *
import datetime
import random



def run_lstm(train_data):

    print("SEQ_LENGTH:", SEQ_LENGTH)
    x_ = tf.placeholder(tf.float32, (None, SEQ_LENGTH, VOCAB_SIZE))
    t_ = tf.placeholder(tf.float32, (None, SEQ_LENGTH, 1))
    y_ = tf.placeholder(tf.float32, (None, SEQ_LENGTH, NUM_CLASSES+1))

    seq_length = tf.placeholder(tf.float32, (None))
    mask_ = tf.placeholder(tf.float32, (None, SEQ_LENGTH))
    inputs = (t_, x_)
    #outputs, _ = dynamic_rnn(cell=tf.contrib.rnn.GRUCell(N_HIDDEN), inputs=x_, dtype=tf.float32, sequence_length=seq_length)
    outputs_pre, _ = dynamic_rnn(cell=PhasedLSTMCell(N_HIDDEN), inputs=inputs, dtype=tf.float32, sequence_length=seq_length)

    from tensorflow.python.ops import rnn, rnn_cell
    cell = rnn_cell.LSTMCell(N_HIDDEN, state_is_tuple=True)
    stacked_cell = rnn_cell.MultiRNNCell([cell] * N_LAYER, state_is_tuple=True)
    outputs, states = rnn.dynamic_rnn(stacked_cell, inputs=outputs_pre, dtype=tf.float32, scope='lstm_layer')

    #fc_dropout = tf.nn.dropout(outputs, 0.5) #??? useful?
    y = slim.fully_connected(inputs=outputs, num_outputs=NUM_CLASSES+1, activation_fn=None)

    '''
    target, prediction = y_, y
    prediction_tmp = prediction + 1e-10
    cross_entropy = target * tf.log(prediction_tmp)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    cross_entropy *= mask_
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(seq_length, tf.float32)
    cross_entropy = tf.reduce_mean(cross_entropy)
    '''

    mask_1 = tf.expand_dims(mask_, axis=2)
    y1 = tf.multiply(y, mask_1)
    y_1 = tf.multiply(y_, mask_1)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y_1))
    grad_update = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    #y_1 =  tf.to_float(tf.argmax(y_, 2), name='ToFloat')
    #y1 = tf.to_float(tf.argmax(y, 2), name='ToFloat')
    #y2 = tf.multiply(y1, mask_)
    #y_2 = tf.multiply(y_1, mask_)
    #correct_prediction = tf.equal(y_2, y2)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    #checkpoint = tf.train.get_checkpoint_state("model/")
    #saver.restore(sess, checkpoint.model_checkpoint_path)
    #NUM_EPOCHS = 5
    #BATCH_SIZE = 2
    leftmost, rightmost = 0, len(train_data['x_'])
    train_data['x_'] = np.array(train_data['x_'])
    train_data['y_'] = np.array(train_data['y_'])
    train_data['mask'] = np.array(train_data['mask'])
    train_data['lengths_'] = np.array(train_data['lengths_'])
    train_data['t_'] = np.array(train_data['t_'])
    tr_acc2 = 0
    print("training_datset_size:"+str(len(train_data['x_'])))

    for epoch in range(NUM_EPOCHS): #NUM_EPOCHS
        batch_number = len(train_data['x_']) // BATCH_SIZE
        y_out, y_temp, mask_temp = 0, 0, 0

        for batch_i in range(batch_number): #batch_number
            randomlist = random.sample(range(leftmost, rightmost), BATCH_SIZE)

            x_temp = train_data['x_'][randomlist]
            y_temp = train_data['y_'][randomlist]
            mask_temp = train_data['mask'][randomlist]
            lengths_temp = train_data['lengths_'][randomlist]
            t_temp = train_data['t_'][randomlist]
            time_temp = np.expand_dims(t_temp, axis=2)
            st = time()
            tr_loss, y_out, _ = sess.run([cross_entropy, y, grad_update], feed_dict={x_:x_temp, t_:time_temp, y_: y_temp, seq_length:lengths_temp, mask_: mask_temp})
            #print('epoch = {0} | steps = {1} | time {2:.2f} | tr_loss = {3:} | tr_acc = {4:} %'.format(str(epoch), str(
            #    batch_i).zfill(6), time() - st, str(tr_loss), str(tr_acc)))

        tr_acc2 = differencce_prediction(y_out, y_temp, mask_temp)

        print('ACC: epoch = {0} || tr_acc = {1:} %'.format(str(epoch), str(tr_acc2)))

        if epoch == NUM_EPOCHS-1 or (epoch != 0 and epoch % 50 == 0):
            save_path = saver.save(sess, "model/phased_LSTM_model_" + str(tr_acc2) +"_"+str(N_HIDDEN)+"_"+str(SEQ_LENGTH)+"_"+str(VOCAB_SIZE)+"_"+str(datetime.datetime.now())+".ckpt")
            print("Model saved in path: %s" % save_path)




def main_fun():
    if args.model == "PLSTM":
        model = PhasedLSTMCell
    else:
        model = globals()[args.model]
    print('Using model = {}'.format(model))
    DATA_ATTR = {
        'max_len': SEQ_LENGTH,
        'debug': DEBUG,
        'source': args.data,
        'with_time': USE_TIME_INFO,
        'with_delta_time': USE_DELTA_TIME
    }
    preprocess_class = Preprocess_Dataset(DATA_ATTR)
    train_data = preprocess_class.load_data()
    train_data = preprocess_class.prepare_data(train_data, VOCAB_SIZE, one_hot=ONE_HOT, sigmoid_on=SIGMOID_ON)
    run_lstm(train_data)



if __name__ == '__main__':
    #main()
    import mnist_dist
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from datetime import datetime
    from tensorflowonspark import TFCluster, TFNode


    sc = SparkContext(conf=SparkConf().setAppName("APP Name"))
    #executors = sc._conf.get("spark.executor.instances")
    #if executors is None:
    #    raise Exception("Could not retrieve the number of executors from the SparkContext")
    #num_executors = int(executors)
    num_ps = 1
    num_executors = 2
    
    cluster = TFCluster.run(sc, main_fun, args, num_executors, num_ps, TFCluster.InputMode.TENSORFLOW)

    images = sc.textFile(args.images).map(lambda ln: [int(x) for x in ln.split(',')])
    labels = sc.textFile(args.labels).map(lambda ln: [float(x) for x in ln.split(',')])
    dataRDD = images.zip(labels)
    # feed data for inference
    prediction_results = cluster.inference(dataRDD)
    prediction_results.take(20)

    cluster.shutdown()


