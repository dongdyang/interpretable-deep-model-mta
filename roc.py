import tensorflow as tf
import scipy.special
import itertools
import random
from collections import defaultdict
from utils import *
from config import *


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            producer_op_list=None
        )
    return graph


def one_embedding_x(input_x, randomlist, lengths_x):
    n_samples = len(randomlist)
    max_len = SEQ_LENGTH
    one_hot_x = np.zeros((n_samples, max_len, VOCAB_SIZE)).astype('int32')
    for i in range(n_samples):
        for j in range(lengths_x[i]):
            '''
            for k in range(2):
                index = randomlist[i]
                kk = input_x[(index, j)][k]
                one_hot_x[i, j, kk] = 1
            '''
            index = randomlist[i]
            one_hot_x[i, j, input_x[(index, j)]] = 1
    x = one_hot_x
    return x

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def differencce(y_out, y_label, length):
    batch_size = len(y_out)
    #y_1 = np.argmax(y_out, 2)
    y_label_temp = np.argmax(y_label, 2)
    ret1, ret2 = [], []
    for batch_ii in range(batch_size):
        for j in range(length[batch_ii]):
            #ret1.extend(y_1[batch_ii][:length[batch_ii]]-1)
            #ret2.extend(y1[batch_ii][:length[batch_ii]]-1)
            position = y_label_temp[batch_ii, j]
            temp = y_label_temp[batch_ii, j] - 1
            ret2.append(temp)
            score = sigmoid(y_out[batch_ii, j, position])
            if temp == 0:
                ret1.append(1.0 - score)
            else:
                ret1.append(score)
    return ret1, ret2



graph = load_graph(frozen_model_filename)
x = graph.get_tensor_by_name('prefix/Placeholder:0')
t = graph.get_tensor_by_name('prefix/Placeholder_1:0')
seq = graph.get_tensor_by_name('prefix/Placeholder_3:0')
prob = graph.get_tensor_by_name('prefix/Placeholder_5:0')
y = graph.get_tensor_by_name('prefix/fully_connected/BiasAdd:0')

file_path = os.path.join("./pkl_10k_"+DATA_DATE+"_train_data.pkl")
max_bytes = 2 ** 31 - 1
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
train_data_pre = pickle.loads(bytes_in)

file_path = os.path.join("./pkl_10k_"+DATA_DATE+"_train_data_done.pkl")
max_bytes = 2 ** 31 - 1
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
train_data = pickle.loads(bytes_in)

train_data['y_'] = np.array(train_data['y_'])
train_data['mask'] = np.array(train_data['mask'])
train_data['lengths_'] = np.array(train_data['lengths_'])
train_data['t_'] = np.array(train_data['t_'])

accuracy = []
y1_total, y2_total = [], []
with tf.Session(graph=graph) as sess:
    batch_number = len(train_data['x_']) // BATCH_SIZE
    for batch_i in range(2):  # batch_number
        mask_temp = train_data['mask'][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        lengths_temp = train_data['lengths_'][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        x_temp = one_embedding_x(train_data['x_'], range(batch_i * BATCH_SIZE, (batch_i + 1) * BATCH_SIZE), lengths_temp) #train_data['x_'][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        y_temp = train_data['y_'][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        t_temp = train_data['t_'][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        time_temp = np.expand_dims(t_temp, axis=2)
        y_out = sess.run(y, feed_dict={x: x_temp, t: time_temp, seq: lengths_temp, prob: 1.0})
        y1, y2 = differencce(y_out, y_temp, lengths_temp)
        y1_total.extend(y1)
        y2_total.extend(y2)

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y2_total, y1_total)
roc_auc = auc(false_positive_rate, true_positive_rate)


plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

print("DONE")

