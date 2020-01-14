
import os
from datetime import datetime
import numpy as np
from config import *
import pickle
from collections import defaultdict


BASE_DIR = os.path.join(os.getcwd(), "data")
USER_RECORD_PATH = 'path.txt'
DELTA_TIME_PATH = ''
ACC_TIME_PATH = 'time_path.txt'

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
        #train_data = [sent for sent in train_data if len(sent) > 1]
        #if self.with_time:
        #    train_time_seq = [delta_time for delta_time in train_time_seq if len(delta_time) > 1]
        '''
        test_data, test_time_seq = load_file(data_source, 'te_', debug)
        train_data = [sent for sent in train_data if len(sent) > 1]
        test_data = [sent for sent in test_data if len(sent) > 2]
        if self.with_time:
            train_time_seq = [delta_time for delta_time in train_time_seq if len(delta_time) > 1]
            test_time_seq = [delta_time for delta_time in test_time_seq if len(delta_time) > 2]
        '''
        # cut data which is too long
        #train_data, train_time_seq = self.cut_reverse_sentences(train_data, self.max_len, train_time_seq)
        #test_data, test_time_seq = cut_sentences(test_data, max_len, test_time_seq)
        #self.check(train_data)

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
            return abs((d2_ - d1_).total_seconds()//300) # every min (same effect if in 1 mins)

        timestamps_aft = []
        for timestamp in timestamps:
            time_temp = int(timestamp_difference(timestamps[0], timestamp))
            timestamps_aft.append(time_temp)
        return timestamps_aft



    def preprocess_timestamp_2version(self, timestamps):
        def timestamp_difference(pretimestampe, aftpretimestampe):
            timeformat = "%Y-%m-%d %H:%M:%S.%f"
            d1_ = datetime.strptime(pretimestampe, timeformat)
            d2_ = datetime.strptime(aftpretimestampe, timeformat)
            return abs((d2_ - d1_).total_seconds()//300) # every min (same effect if in 1 mins)

        timestamps_aft = []
        for timestamp in timestamps:
            time_temp = int(timestamp_difference(timestamps[0], timestamp))
            timestamps_aft.append(time_temp)
        return timestamps_aft




    def preprocess_channelID(self, item_seq):
        #number2name = { "2": 1, "25": 2, "6": 3, "16": 4, "9": 5, "23": 6, "26": 7, "15": 8, "3": 9, "14": 10, "17": 11, "32": 12,
        #    "19": 13, "27": 14, "33": 15, "13": 16, "1": 17, "28": 18, "18": 19, "20": 20, "30": 21}
        item_seq_res = []
        for ele in item_seq:
            #item_seq_res.append(int(ele))
            item_seq_res.append(ele)
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
                    #item_seq = self.preprocess_channelID(item_seq)
                    sentences.append(item_seq)
                    count += 1
                    # use a small subset if debug on
                    if debug and count == 50:
                        break
        else:
            print("error in reading file")
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


    def process_one_hot_input(self, ele, rotationMap):
        res = []
        channel, rotation = ele.split(",")
        if rotation not in rotationMap:
            rotationMap[rotation] = 50 + len(rotationMap)

        return [int(channel), rotationMap[rotation]]

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

        #x = np.zeros((n_samples, max_len)).astype('int32')
        x = defaultdict(list)
        t = np.zeros((n_samples, max_len)).astype('float')

        rotationMap = defaultdict()
        mask = np.zeros((n_samples,max_len)).astype('float')
        for idx, sent in enumerate(x_origin):
            for idy in range(lengths_x[idx]):
                x[(idx, idy)] = self.process_one_hot_input(sent[idy], rotationMap)
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

        '''
        if one_hot:
            one_hot_x = np.zeros((n_samples, max_len, vocab_size)).astype('int32')
            one_hot_y = np.zeros((n_samples, max_len, NUM_CLASSES+1)).astype('int32')

            for i in range(n_samples):
                for j in range(lengths_x[i]):
                    for k in range(2):
                        kk = x[(i, j)][k]
                        one_hot_x[i, j, kk] = 1
                    one_hot_y[i, j, y[i, j]] = 1

            x = one_hot_x
            y = one_hot_y
        else:
            print("Error, should be one-hot embeddings")
            #x = x.reshape(x.shape[0], x.shape[1], ndim)

        '''

        if one_hot:
            one_hot_y = np.zeros((n_samples, max_len, NUM_CLASSES + 1)).astype('int32')

            for i in range(n_samples):
                for j in range(lengths_x[i]):
                    one_hot_y[i, j, y[i, j]] = 1

            y = one_hot_y
        else:
            print("Error, should be one-hot embeddings")
            # x = x.reshape(x.shape[0], x.shape[1], ndim)

        ret = {'x_':x, 'y_':y, 'mask':mask, 'lengths_':lengths_x}
        if t_origin is not None:
            ret['t_'] = t

        return ret



    def generate_path(self):

        '''
        *0  mpx_chnl_id smallint,
        *7  roi_event_ts timestamp,
        *8  roi_guid string,
        *9  roi_cguid string,
        *10  user_id decimal(18,0),
        *11  trans_type_cd int,
        *12  transaction_type string,
        *15  roi_rotation_id decimal(18,0),
        *17  click_rvr_id decimal(18,0),
        *19  click_event_ts timestamp,
        *22  click_brwsr_name string,
        *23  click_rotation_id decimal(20,0),
        *25  click_guid string,
        *26  click_cguid string,
        *35  click_event_rank int,
        '''
        print("begin to generate path")

        date = "2018-07-13"
        clean_file = "../roi_event_dt=" + date
        path_dict = {}
        #file = open(clean_file)
        rotationMap = defaultdict()

        files = os.listdir(clean_file)

        for file_name in files:
            file = open(os.path.join(clean_file, file_name))

            while True:
                line = file.readline()
                if not line:
                    break

                context = line.split("\t")
                channel_id = context[0].replace("\n", "")
                use_id = context[1].replace("\n", "")
                click_event_ts = context[2].replace("\n", "")
                rotation_id = context[3].replace("\n", "")
                click_event_rank = context[4].replace("\n", "")
                #gmb = context[5].replace("\n", "")

                rotationMap[rotation_id] = 1

                if use_id not in path_dict:
                    path_dict[use_id] = {}
                if click_event_ts not in path_dict[use_id]:
                    path_dict[use_id][click_event_ts] = [click_event_rank, channel_id, rotation_id, gmb]

            file.close()

        print("length of rotation: "+ str(len(rotationMap)))

        print("generate path mid-DONE")

        file_gmb_path1 = "./data/roi_event_dt=" + date + "/path.txt"
        file1 = open(file_gmb_path1, 'w')

        file_gmb_path2 = "./data/roi_event_dt=" + date + "/time_path.txt"
        file2 = open(file_gmb_path2, 'w')

        file_gmb = "./data/roi_event_dt=" + date + "/gmb.txt"
        file3 = open(file_gmb, 'w')

        for ele in path_dict:
            temp = path_dict[ele]
            click_event_tss = sorted(path_dict[ele])
            path = []
            time_path = []

            for click_event_ts in click_event_tss:
                click_event_rank = temp[click_event_ts][0]
                channel_id = temp[click_event_ts][1]
                rotation_id = temp[click_event_ts][2]
                #gmb = temp[click_event_ts][3]

                path.append(channel_id + "," + rotation_id)
                time_path.append(click_event_ts)

                if int(click_event_rank) == 1:
                    if len(path) == 1:
                        continue
                    file1.write("\t".join(path[-32:]) + "\n")
                    file2.write("\t".join(time_path[-32:]) + "\n")
                    #file3.write(str(gmb)+"\n")

                    path = []
                    time_path = []

        file1.close()
        file2.close()
        print("generate path DONE")


    def read_data(self, file_path):
        max_bytes = 2 ** 31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        train_data = pickle.loads(bytes_in)
        return train_data


if __name__ == '__main__':
    DATA_ATTR = {
        'max_len': SEQ_LENGTH,
        'source': DATA_DATE,
        'with_time': USE_TIME_INFO,
        'with_delta_time': USE_DELTA_TIME
    }
    preprocess_class = Preprocess_Dataset(DATA_ATTR)

    preprocess_class.generate_path()
    '''
    print("Step 1")
    train_data = preprocess_class.load_data()
    print("Step 2")
    train_data_done = preprocess_class.prepare_data(train_data, VOCAB_SIZE, one_hot=ONE_HOT)
    print("Step 3")


    file_path = "pkl_10k_"+DATA_DATE+"_train_data.pkl"
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(train_data, protocol=4)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


    file_path = "pkl_10k_"+DATA_DATE+"_train_data_done.pkl"
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(train_data_done, protocol=4)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])
    '''




