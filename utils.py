
import os
import random
from datetime import datetime
import numpy as np
from config import *

BASE_DIR = os.path.join(os.getcwd(), "data")
USER_RECORD_PATH = 'path.txt'
DELTA_TIME_PATH = ''
ACC_TIME_PATH = 'time_path.txt'


class Preprocess_Dataset:
    def __init__(self, data_attr):
        # max_len = 0,# Max length of setence
        # vocab_size = 0, # vocabulary size
        # debug=False, # return a small set if True
        # val_num=100,  # number of validation sample
        # with_time=False, # return time information
        # with_delta_time=False # return delta time if True else if with_time == True return time
        # return: two dictionary
        # train = {'x':..., 'y':..., 't':...}
        # test = {'x':..., 'y':..., 't':...}
        self.max_len = data_attr.get('max_len', 50)
        self.debug = data_attr.get('debug', False)
        self.with_time = data_attr.get('with_time', False)
        self.with_delta_time = data_attr.get('with_delta_time', False)
        self.data_source = data_attr.get('source', '2018-04-25')
        self.with_time = self.with_time or self.with_delta_time


    def load_data(self):
        train_data, train_time_seq = self.load_file(self.data_source, self.debug)
        train_data = [sent for sent in train_data if len(sent) > 1]
        if self.with_time:
            train_time_seq = [delta_time for delta_time in train_time_seq if len(delta_time) > 1]
        '''
        test_data, test_time_seq = load_file(data_source, 'te_', debug)
        train_data = [sent for sent in train_data if len(sent) > 1]
        test_data = [sent for sent in test_data if len(sent) > 2]
        if self.with_time:
            train_time_seq = [delta_time for delta_time in train_time_seq if len(delta_time) > 1]
            test_time_seq = [delta_time for delta_time in test_time_seq if len(delta_time) > 2]
        '''
        # cut data which is too long
        train_data, train_time_seq = self.cut_reverse_sentences(train_data, self.max_len, train_time_seq)
        #test_data, test_time_seq = cut_sentences(test_data, max_len, test_time_seq)
        self.check(train_data)

        ytr = [[1]*(len(train_data[index])-1)+[2] for index in range(len(train_data))]   ###########     This one generate the label
        xtr, ttr = train_data, train_time_seq

        train = {'x_': xtr, 'y_': ytr}
        #test = {'x': xte, 'y': yte}
        if self.with_time:
            train['t_'] = ttr
            #test['t'] = te

        return train#, test



    def preprocess_timestamp(self, timestamps):
        def timestamp_difference(pretimestampe, aftpretimestampe):
            timeformat = "%Y-%m-%d %H:%M:%S.%f"
            timeformat2 = "%Y-%m-%d %H:%M:%S"
            try:
                d1_ = datetime.strptime(pretimestampe, timeformat)
            except:
                d1_ = datetime.strptime(pretimestampe, timeformat2)
            try:
                d2_ = datetime.strptime(aftpretimestampe, timeformat)
            except:
                d2_ = datetime.strptime(aftpretimestampe, timeformat2)
            return abs((d2_ - d1_).total_seconds()//180) # every min (same effect if in 1 mins)

        timestamps_aft = []
        for timestamp in timestamps:
            time_temp = int(timestamp_difference(timestamps[0], timestamp))
            timestamps_aft.append(time_temp)
        return timestamps_aft


    def preprocess_channelID(self, item_seq):
        number2name = { "2": 1, "25": 2, "6": 3, "16": 4, "9": 5, "23": 6, "26": 7, "15": 8, "3": 9, "14": 10, "17": 11, "32": 12,
            "19": 13, "27": 14, "33": 15, "13": 16, "1": 17, "28": 18, "18": 19, "20": 20, "30": 21}
        item_seq_res = []
        for ele in item_seq:
            #item_seq_res.append(int(ele))
            item_seq_res.append(number2name[ele])
        return item_seq_res

    def load_file(self, data_source, debug=False):
        sentences = []
        user_record_path = os.path.join(BASE_DIR, "roi_event_dt=" + data_source, USER_RECORD_PATH)

        if os.path.exists(user_record_path):
            with open(user_record_path, 'r') as f:
                count = 0
                for line in f:
                    line_context = line.strip().split('\t')
                    item_seq = line_context
                    item_seq = self.preprocess_channelID(item_seq)
                    sentences.append(item_seq)
                    count += 1
                    # use a small subset if debug on
                    if debug and count == 50:
                        break
        else:
            exit()

        time_seq = None
        time_file_path = os.path.join(BASE_DIR, "roi_event_dt=" + data_source, ACC_TIME_PATH)
        if self.with_delta_time:
            time_file_path = os.path.join(BASE_DIR, "roi_event_dt=" + data_source, DELTA_TIME_PATH)

        if self.with_time and os.path.exists(time_file_path):
            time_seq = []
            with open(time_file_path, 'r') as f:
                count = 0
                for line in f:
                    line_context = line.strip().split('\t')
                    delta = line_context

                    delta = self.preprocess_timestamp(delta)

                    if len(delta) != len(sentences[count]):
                        del sentences[count]
                        continue
                    time_seq.append(delta)
                    count += 1
                    if debug and count == 50:
                        break
        elif self.with_time:
            print("ERROR self.with_time, File utils.py")
            exit()
        return sentences, time_seq


    def cut_reverse_sentences(self, sentences, max_len, time_seq=None):
        # remove the sentences: len < 2 and len > max_len
        dt_ret = None
        '''
        # reverse part. Throw. Make No sense Actually.
        sentences = []
        time_seq = []
        for i in range(len(sentences_pre)):
            sentences.append(sentences_pre[i][::-1])
            time_seq.append(time_seq_pre[i][::-1])
        '''
        if max_len:
            sents_ret = [sent[-max_len:] for sent in sentences]
            if time_seq is not None:
                dt_ret = [delta_time[-max_len:] for delta_time in time_seq]
        else:
            sents_ret = sentences
            if time_seq is not None:
                dt_ret = time_seq

        return sents_ret, dt_ret


    def check(self, sentences, time_seq=None):
        # show the data statics
        max_len = 0
        total = 0
        lengths = []
        if time_seq is not None:
            for delta_time, sent in zip(time_seq, sentences):
                assert (len(delta_time) == len(sent))
        for sent in sentences:
            length = len(sent)
            lengths.append(length)
            total += length
            max_len = max_len if max_len > length else length


    def prepare_data(self, data, vocab_size, one_hot=False, sigmoid_on=False):
        '''
        convert list of data into numpy.array
        padding 0
        generate mask
        '''
        def sigmoid(x):
            # return sigmoid on x
            x = np.array(x)
            out = 1. / (1 + np.exp(-x))
            return out

        x_origin = data['x_']
        y_origin = data['y_']
        t_origin = data.get('t_', None)
        ndim = 1 if not one_hot else vocab_size

        lengths_x = [len(s) for s in x_origin]
        n_samples = len(x_origin)
        max_len = SEQ_LENGTH #np.max(lengths_x) #

        x = np.zeros((n_samples, max_len)).astype('int32')
        t = np.zeros((n_samples, max_len)).astype('float')

        mask = np.zeros((n_samples,max_len)).astype('float')
        for idx, sent in enumerate(x_origin):
            x[idx, :lengths_x[idx]] = sent
            mask[idx, :lengths_x[idx]] = 1.
            if t_origin:
                tmp_t = t_origin[idx]
                if sigmoid_on:
                    tmp_t = sigmoid(tmp_t)
                t[idx,:int(np.sum(mask[idx]))] = tmp_t

        if type(y_origin[0]) is list:
            # train
            y = np.zeros((n_samples, max_len)).astype('int32')
            lengths_y = [len(s) for s in y_origin]
            for idx, sent in enumerate(y_origin):
                y[idx, :lengths_y[idx]] = sent
        else:
            # test
            y = np.array(y_origin).astype('int32')

        if one_hot:
            one_hot_x = np.zeros((n_samples, max_len, vocab_size)).astype('int32')
            one_hot_y = np.zeros((n_samples, max_len, NUM_CLASSES+1)).astype('int32')

            for i in range(n_samples):
                for j in range(max_len):
                    one_hot_x[i, j, x[i, j]] = 1
                    one_hot_y[i, j, y[i, j]] = 1

            x = one_hot_x
            y = one_hot_y
        else:
            x = x.reshape(x.shape[0], x.shape[1], ndim)

        ret = {'x_':x,'y_':y, 'mask':mask, 'lengths_':lengths_x}
        if t_origin is not None:
            ret['t_'] = t

        return ret

def differencce_prediction(y_out, y_temp, mask_temp, return_type=False):
    batch_size = len(y_out)
    y_1 = np.argmax(y_out, 2)
    y1 = np.argmax(y_temp, 2)
    y2 = np.multiply(y1, mask_temp)
    y_2 = np.multiply(y_1, mask_temp)
    accuracy2 = 0
    accuracy2_shape = []
    for batch_ii in range(batch_size):
        correct_prediction = 0
        total_prediction = 0
        for index, ele in enumerate(mask_temp[batch_ii]):
            if ele:
                if y2[batch_ii][index] == y_2[batch_ii][index]:
                    correct_prediction += 1
                total_prediction += 1
        if return_type:
            accuracy2_shape.append(float(correct_prediction * 100 / total_prediction))
        else:
            accuracy2 += float(correct_prediction * 100 / total_prediction)
    tr_acc2 = float(accuracy2 / batch_size)
    return tr_acc2 if not return_type else accuracy2_shape