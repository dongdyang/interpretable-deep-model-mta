import argparse

parser = argparse.ArgumentParser(description='Specific model, data and other params.')
parser.add_argument('--model', type=str, default='PLSTM', help='Model to train:LSTM, PLSTM.')
parser.add_argument('--data', type=str, default='2018-04-25', help='Input data source: roi_event_dt')
parser.add_argument('--num_classes', type=int, default=2, help='Input data number of labels')
parser.add_argument('--fixed_epochs', type=int, default=5, help='Number of epochs in the first stage.')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs in the first and second stage.')
parser.add_argument('--num_hidden', type=int, default=32, help='Number of hidden unit.')
#parser.add_argument('--num_hidden_rnn', type=int, default=256, help='Number of hidden unit.')
parser.add_argument('--num_layers', type=int, default=2, help='Batch size in the training phase.')

parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--sample_time', type=int, default=3, help='Sample time in the evaluate method.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size in the training phase.')
parser.add_argument('--test_batch', type=int, default=5, help='Batch size in the testing phase')
parser.add_argument('--vocab_size', type=int, default=32, help='Vocabulary size')
parser.add_argument('--max_len', type=int, default=32, help='Maximum length of the sequence.')
parser.add_argument('--one_hot', type=bool, default=True, help='one hot embedding.')
parser.add_argument('--grad_clip', type=int, default=0, help='Maximum grad step. Grad will be cliped if greater than this. 0 means no clip')
parser.add_argument('--debug', dest='debug', action='store_true', help='If debug is set, train one time, load small dataset.')
parser.add_argument('--bn', dest='bn', action='store_true', help='If bn is set, input data will be batch normed')
parser.add_argument('--sigmoid_on', dest='sigmoid_on', action='store_true', help='if sigmoid_on is set, input time data will be sigmoid')
parser.add_argument("--model_folder", type=str, default="model", help="Model folder to export")
parser.add_argument("--frozen_model_filename", default="model/frozen_model.pb", type=str, help="Frozen model file to import")

parser.set_defaults(debug=False)
parser.set_defaults(sigmoid_on=False)
parser.set_defaults(bn=False)
args = parser.parse_args()
#######################################################
DEBUG = args.debug
SIGMOID_ON = args.sigmoid_on
BN = args.bn
DATA_DATE = args.data
NUM_CLASSES = args.num_classes
SEQ_LENGTH = args.max_len
VOCAB_SIZE = args.vocab_size
MODEL_TYPE = args.model
N_LAYER = args.num_layers
N_HIDDEN = args.num_hidden
#N_HIDDEN_RNN = args.num_hidden_rnn
LEARNING_RATE = args.learning_rate
GRAD_CLIP = args.grad_clip
NUM_EPOCHS = args.num_epochs
FIXED_EPOCHS = args.fixed_epochs
BATCH_SIZE = args.batch_size
TEST_BATCH = args.test_batch
SAMPLE_TIME = args.sample_time
PRINT_FREQ = 20
ONE_HOT = args.one_hot
if DEBUG:
    PRINT_FREQ = 1
USE_TIME_INPUT = False
USE_TIME_INFO = False
USE_DELTA_TIME = False
if MODEL_TYPE == 'PLSTM' :
    USE_TIME_INPUT = True
    USE_TIME_INFO = True
elif MODEL_TYPE == 'LSTM':
    pass
else:
    print("Wrong Modle specified {}".format(MODEL_TYPE))
    exit()