import argparse

import tensorflow as tf
from utils import Preprocess_Dataset
from config import *
import utils
import scipy.special
import numpy as np
import itertools

import random
from collections import defaultdict
from utils import *



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

graph = load_graph(args.frozen_model_filename)
x = graph.get_tensor_by_name('prefix/Placeholder:0')
t = graph.get_tensor_by_name('prefix/Placeholder_1:0')
seq = graph.get_tensor_by_name('prefix/Placeholder_3:0')
y = graph.get_tensor_by_name('prefix/fully_connected/BiasAdd:0')


def infer(single_data):
    # graph = load_graph(args.frozen_model_filename)
    #for op in graph.get_operations():
    #    print(op.name)

    with tf.Session(graph=graph) as sess:
        x_temp = np.array(single_data["x_"])
        sequence_length = np.array(single_data["lengths_"])
        time_temp = np.array(single_data["t_"])
        y_out = sess.run(y, feed_dict={x: x_temp, t: time_temp, seq: sequence_length})
        return y_out


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def shapley_kernel(M,s):
    if s == 0 or s == M:
        return 10000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))


def kernel_shap(f, x, reference, M):
    X = np.zeros((2**M,M+1))
    X_MASK = np.zeros((SEQ_LENGTH, 1))
    X[:,-1] = 1
    weights = np.zeros(2**M)
    V = defaultdict()
    V['lengths_'] = []
    V['t_'] = []
    V['x_'] = []
    V['y_'] = []
    V['mask'] = []
    #ws = {}
    powerset_value = powerset(range(M))
    for i,s in enumerate(powerset_value):
        s = list(s)
        #print(s)
        #V[i,s] = x[s]
        X[i, s] = 1
        X_MASK[s] = 1

        #watch_var = np.multiply(x['lengths_'], X_MASK)
        V['lengths_'].append(x['lengths_'])
        V['t_'].append(np.multiply(x['t_'], X_MASK))
        V['x_'].append(np.multiply(x['x_'], X_MASK))
        V['y_'].append(x['y_'])
        V['mask'].append(x['mask'])
        #ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(M,len(s))
        weights[i] = shapley_kernel(M,len(s))
        X_MASK[s] = 0

    y_output = np.array(f(V))
    acc = differencce_prediction(y_output, V['y_'], V['mask'], True)

    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), acc))


# The followings are for only ONE data.
def one_small_shapley(train_data, index=0):
    lengths_temp = train_data['lengths_'][index]
    x_temp = train_data['x_'][index]
    mask_temp = train_data['mask'][index]
    y_temp = train_data['y_'][index]
    time_temp = np.expand_dims(train_data['t_'][index], axis=2)

    M = lengths_temp
    reference = np.zeros((M, VOCAB_SIZE))
    single_data = {"x_": x_temp, "lengths_": lengths_temp, "t_": time_temp, "y_": y_temp, "mask": mask_temp}

    phi = kernel_shap(infer, single_data, reference, M)
    base_value = phi[-1]
    shap_values = phi[:-1]

    return shap_values


def kernel_shap_sampling(f, x, reference, M, sample_number):

    powerset_number = 2**M
    X = np.zeros((sample_number*2*M,M+1))
    X_MASK = np.zeros((SEQ_LENGTH, 1))
    X[:,-1] = 1

    i = 0
    V = defaultdict()
    V['lengths_'], V['t_'], V['x_'] = [], [], []
    V['mask'] = []
    V['y_'] = []
    #acc1, acc2 = 0, 0
    #feature_importance = np.zeros(M+1)

    weights = np.zeros(sample_number*M*2)
    for feature_i in range(M):
        for _ in range(sample_number):
            binary_combination_tmp = random.randint(0, powerset_number - 1)
            binary_combination = list(str("{0:b}".format(binary_combination_tmp)).zfill(M))

            # Not contain feature i
            s = [j for j, e in enumerate(binary_combination) if e != '0']
            #print(binary_combination_tmp, s)
            X[i, s] = 1
            X_MASK[s] = 1
            V['lengths_'].append(x['lengths_'])
            V['t_'].append(np.multiply(x['t_'], X_MASK))
            V['x_'].append(np.multiply(x['x_'], X_MASK))
            V['mask'].append(x['mask'])
            V['y_'].append(x['y_'])
            X_MASK[s] = 0
            weights[i] = shapley_kernel(M, len(s))
            i += 1

            # Contain feature i
            binary_combination[feature_i] = 1
            s = [j for j, e in enumerate(binary_combination) if e != '0']
            X[i, s] = 1
            X_MASK[s] = 1
            V['lengths_'].append(x['lengths_'])
            V['t_'].append(np.multiply(x['t_'], X_MASK))
            V['x_'].append(np.multiply(x['x_'], X_MASK))
            V['mask'].append(x['mask'])
            V['y_'].append(x['y_'])
            X_MASK[s] = 0
            weights[i] = shapley_kernel(M, len(s))
            i += 1

        #y = np.array(f(V))
        #acc1 = differencce_prediction(y, V['y_'], V['mask'])
        #y_contain_i = np.array(f(V_contain_i))
        #acc2 = differencce_prediction(y_contain_i, V_contain_i['y_'], V_contain_i['mask'])
        #feature_importance[feature_i] = float((acc2 - acc1))
    #return feature_importance

    y_output = np.array(f(V))
    acc = differencce_prediction(y_output, V['y_'], V['mask'], True)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), acc))


def one_big_shapley(train_data, index=0):
    lengths_temp = train_data['lengths_'][index]
    if lengths_temp <= 7:
        return one_small_shapley(train_data, index)

    x_temp = train_data['x_'][index]
    mask_temp = train_data['mask'][index]
    y_temp = train_data['y_'][index]
    time_temp = np.expand_dims(train_data['t_'][index], axis=2)

    M = lengths_temp
    reference = np.zeros((M, VOCAB_SIZE))
    single_data = {"x_": x_temp, "lengths_": lengths_temp, "t_": time_temp, "y_": y_temp, "mask": mask_temp}

    sample_number = np.math.factorial(4)
    phi = kernel_shap_sampling(infer, single_data, reference, M, sample_number)
    base_value = phi[-1]
    shap_values = phi[:-1]
    return shap_values



def cal_shape_value():
    DATA_ATTR = {
        'max_len': SEQ_LENGTH,
        'debug': DEBUG,
        'source': DATA_DATE,
        'with_time': USE_TIME_INFO,
        'with_delta_time': USE_DELTA_TIME
    }
    preprocess_class = Preprocess_Dataset(DATA_ATTR)
    train_data = preprocess_class.load_data()
    train_data_done = preprocess_class.prepare_data(train_data, VOCAB_SIZE, one_hot=ONE_HOT, sigmoid_on=SIGMOID_ON)


    weight_all = defaultdict(float)
    occurance_number = defaultdict(float) # for user journey

    for test_index in range(3800):
        #one_small_shapley(train_data_done, test_index)
        shap_values = one_big_shapley(train_data_done, test_index)

        sample = train_data['x_'][test_index]
        sample_set = set(sample)
        print(test_index, sample)

        shap_values = [ele if ele > 0 else 0 for ele in shap_values]
        total = sum(shap_values)
        if total != 0:
            shap_values /= total
        for event_i in range(len(sample)):
            watch = sample[event_i]
            watch2 = shap_values[event_i]
            weight_all[sample[event_i]] += shap_values[event_i]
        for ele in sample_set:
            occurance_number[ele] += 1

        #print("Shapley Value:"+str(shap_values))

    for channle in weight_all:
        weight_all[channle] /= occurance_number[channle]

    total = sum(weight_all.values())
    weight_all_new = defaultdict(float)
    number2name = {1: "2", 2: "25", 3: "6", 4: "16", 5: "9", 6: "23", 7: "26", 8: "15", 9: "3", 10: "14", 11: "17",
                   12: "32", 13: "19", 14: "27", 15: "33", 16: "13", 17: "1", 18: "28", 19: "18", 20: "20", 21: "30"}

    if total != 0:
        for ele in weight_all:
            weight_all_new[number2name[ele]] = weight_all[ele] / total

    print(weight_all_new)


def infer_large_dataset(train_data):
    #for op in graph.get_operations():
    #   print(op.name)


    batch_size_temp = 0
    if len(train_data['x_']) >= 50:
        batch_size_temp = 50
    else:
        batch_size_temp = 1

    accuracy = []
    with tf.Session(graph=graph) as sess:
        batch_number = len(train_data['x_']) // batch_size_temp
        for batch_i in range(batch_number):  # batch_number
            mask_temp = train_data['mask'][batch_i * batch_size_temp:(batch_i + 1) * batch_size_temp]
            x_temp = train_data['x_'][batch_i * batch_size_temp:(batch_i + 1) * batch_size_temp]
            y_temp = train_data['y_'][batch_i * batch_size_temp:(batch_i + 1) * batch_size_temp]
            lengths_temp = train_data['lengths_'][batch_i * batch_size_temp:(batch_i + 1) * batch_size_temp]
            t_temp = train_data['t_'][batch_i * batch_size_temp:(batch_i + 1) * batch_size_temp]
            time_temp = np.expand_dims(t_temp, axis=2)
            y_out = sess.run(y, feed_dict={x: x_temp, t: time_temp, seq: lengths_temp})
            accuracy.append(differencce_prediction(y_out, y_temp, mask_temp))

    return sum(accuracy) / len(accuracy)



def kernel_simplediff(raw_train_data, channel_data, channel_id):
    acc1 = infer_large_dataset(channel_data)

    new_channel_data = defaultdict()
    new_channel_data['lengths_'] = np.zeros(len(raw_train_data))
    new_channel_data['t_'], new_channel_data['x_'] = [], []
    new_channel_data['mask'] = []
    new_channel_data['y_'] = []

    new_channel_data_index = -1

    for path_index, path in enumerate(raw_train_data):
        #new_channel_data['lengths_'].append([])
        new_channel_data['t_'].append([])
        new_channel_data['x_'].append([])
        new_channel_data['mask'].append([])
        new_channel_data['y_'].append([])
        new_channel_data_index += 1

        #for path_ele_index, path_ele in enumerate(path):
        for path_ele_index in range(SEQ_LENGTH):
            path_ele = path[path_ele_index] if path_ele_index < len(path) else 0

            mask = 1
            if path_ele ==  channel_id:
                mask = 0

            #watch = channel_data['t_'][path_index][path_ele_index]
            #watch2 = np.multiply(watch, mask)

            new_channel_data['t_'][new_channel_data_index].append(np.multiply(channel_data['t_'][path_index][path_ele_index], mask))
            new_channel_data['x_'][new_channel_data_index].append(np.multiply(channel_data['x_'][path_index][path_ele_index], mask))
            new_channel_data['mask'][new_channel_data_index].append(channel_data['mask'][path_index][path_ele_index])
            new_channel_data['y_'][new_channel_data_index].append(channel_data['y_'][path_index][path_ele_index])

        new_channel_data['lengths_'][new_channel_data_index] = channel_data['lengths_'][path_index]

    new_channel_data['x_'] = np.array(new_channel_data['x_'])
    new_channel_data['y_'] = np.array(new_channel_data['y_'])
    new_channel_data['mask'] = np.array(new_channel_data['mask'])
    new_channel_data['lengths_'] = np.array(new_channel_data['lengths_'])
    new_channel_data['t_'] = np.array(new_channel_data['t_'])


    acc2 = infer_large_dataset(new_channel_data)
    print(str(channel_id)+"\tBefore:"+str(acc1)+"\tAfter:"+str(acc2))
    return acc1 - acc2





def simple_prediction_diff(raw_train_data, train_data, channel_contain):
    weights = defaultdict()

    raw_train_data['x_'] = np.array(raw_train_data['x_'])
    train_data['x_'] = np.array(train_data['x_'])
    train_data['y_'] = np.array(train_data['y_'])
    train_data['mask'] = np.array(train_data['mask'])
    train_data['lengths_'] = np.array(train_data['lengths_'])
    train_data['t_'] = np.array(train_data['t_'])

    for channel_id in channel_contain:
        raw_x_temp = raw_train_data['x_'][channel_contain[channel_id]]

        lengths_temp = train_data['lengths_'][channel_contain[channel_id]]
        x_temp = train_data['x_'][channel_contain[channel_id]]
        mask_temp = train_data['mask'][channel_contain[channel_id]]
        y_temp = train_data['y_'][channel_contain[channel_id]]
        time_temp = train_data['t_'][channel_contain[channel_id]]

        channel_data = {"x_": x_temp, "lengths_": lengths_temp, "t_": time_temp, "y_": y_temp, "mask": mask_temp}

        weight = kernel_simplediff(raw_x_temp, channel_data, channel_id)
        weights[channel_id] = weight

    total = sum(weights.values())
    weight_all_new = defaultdict(float)
    number2name = {1: "2", 2: "25", 3: "6", 4: "16", 5: "9", 6: "23", 7: "26", 8: "15", 9: "3", 10: "14", 11: "17",
                   12: "32", 13: "19", 14: "27", 15: "33", 16: "13", 17: "1", 18: "28", 19: "18", 20: "20", 21: "30"}
    if total != 0:
        for ele in weights:
            weight_all_new[number2name[ele]] = weights[ele] / total

    return weight_all_new



def simply_calculation():
    DATA_ATTR = {
        'max_len': SEQ_LENGTH,
        'debug': DEBUG,
        'source': DATA_DATE,
        'with_time': USE_TIME_INFO,
        'with_delta_time': USE_DELTA_TIME
    }
    preprocess_class = Preprocess_Dataset(DATA_ATTR)
    train_data = preprocess_class.load_data()
    train_data_done = preprocess_class.prepare_data(train_data, VOCAB_SIZE, one_hot=ONE_HOT, sigmoid_on=SIGMOID_ON)

    channel_contain = defaultdict()
    for path_index, path in enumerate(train_data["x_"]):
        path_set = set(path)
        for ele in path_set:
            if ele not in channel_contain:
                channel_contain[ele] = []
            channel_contain[ele].append(path_index)

    weigths = simple_prediction_diff(train_data, train_data_done, channel_contain)

    print(weigths)



if __name__ == '__main__':
    # infer_large_dataset()

    from time import time
    pre = time()
    cal_shape_value()
    #simply_calculation()
    print(time()-pre)


