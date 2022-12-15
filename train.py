import numpy as np
import os
import sys
import random
import optparse
import keras
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import Average
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import h5py
import sklearn
from sklearn.metrics import roc_auc_score 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
channel_num = 4
cur_filename = os.path.split(sys.argv[0])[1]
prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-l", "--len", action = "store", type = int, dest = "cutoff_len",
									help = "cutoff_len")
parser.add_option("-i", "--intr", action = "store", type = "string", dest = "Train_dir_in",
									default='./', help = "input directory for training data")
parser.add_option("-j", "--inval", action = "store", type = "string", dest = "Valid_dir_in",
									default='./', help = "input directory for validation data")
parser.add_option("-o", "--out", action = "store", type = "string", dest = "Result_dir_out",
									default='./', help = "output directory")
parser.add_option("-f", "--fLen", action = "store", type = int, dest = "filter_len",
									default=10, help = "the length of filter")
parser.add_option("-n", "--fNum", action = "store", type = int, dest = "filter_num",
									default=500, help = "number of filters in the convolutional layer")
parser.add_option("-d", "--dense", action = "store", type = int, dest = "dense_num",
									default=500, help = "number of neurons in the dense layer")
parser.add_option("-e", "--epochs", action = "store", type = int, dest = "epoch",
									default=10, help = "number of epochs")
parser.add_option("-w", "--train?", action = "store", type = int, dest = "is_train",
									default="Y", help = "Judge weather train")

(option,args) = parser.parse_args()
if (option.cutoff_len is None or option.filter_num is None or option.dense_num is None or option.epoch is None):
    sys.stderr.write("-------------------------------------------------------------------------------\n" +
    cur_filename + " appear ERROR: missing some required command argument" + 
                "\n-------------------------------------------------------------------------------\n")
    parser.print_help()
    sys.exit(0)
else:
	print("---------------------------------------------------------------------")
	print("|Initialize required argument:"+
	"\n|cutoff_len is           : " + str(option.cutoff_len) +
	"\n|Train_dir_in is         : " + option.Train_dir_in +
	"\n|Valid_dir_in is         : " + option.Valid_dir_in +
	"\n|Result_dir_out is       : " + option.Result_dir_out +
	"\n|filter_len is           : " + str(option.filter_len) +
	"\n|filter_num is           : " + str(option.filter_num) +
	"\n|dense_num is            : " + str(option.dense_num) +
	"\n|epoch is                : " + str(option.epoch))
	print("---------------------------------------------------------------------")


cutoff_len = option.cutoff_len
cutoff_lenk = str(cutoff_len/1000)
filter_len = option.filter_len
filter_num = option.filter_num
dense_num = option.dense_num
epoch = option.epoch
Train_dir_in = option.Train_dir_in
Valid_dir_in = option.Valid_dir_in
Result_dir_out = option.Result_dir_out
is_train = option.is_train
if not os.path.exists(Result_dir_out):
	os.makedirs(Result_dir_out)
print("......\nThe model store directory has been created at:" + ' ' + Result_dir_out)

print("......\nStarting loading data from train set and validation set......")

print("......\nLoading virus sequence......")
print(Train_dir_in)
tr_virus_data_filename = [f for f in os.listdir(Train_dir_in) if 'codefw.npy' in f and 'virus' in f and str(cutoff_len)+'_' in f][0]
tr_virus_data_r_filename = [f for f in os.listdir(Train_dir_in) if 'codebw.npy' in f and 'virus' in f and str(cutoff_len)+'_' in f][0]
print("data for training from -> " + tr_virus_data_filename + "  and  " + tr_virus_data_r_filename)
tr_virus_data = np.load(os.path.join(Train_dir_in,tr_virus_data_filename))
tr_virus_data_r = np.load(os.path.join(Train_dir_in,tr_virus_data_r_filename))

val_virus_data_filename = [f for f in os.listdir(Valid_dir_in) if 'codefw.npy' in f and 'virus' in f and str(cutoff_len)+'_' in f][0]
val_virus_data_r_filename = [f for f in os.listdir(Valid_dir_in) if 'codebw.npy' in f and 'virus' in f and str(cutoff_len)+'_' in f][0]
print("data for validating from -> " + val_virus_data_filename + "  and  " + val_virus_data_r_filename)
val_virus_data = np.load(os.path.join(Valid_dir_in,val_virus_data_filename))
val_virus_data_r = np.load(os.path.join(Valid_dir_in,val_virus_data_r_filename))

print("......\nLoading host sequence......")

tr_host_data_filename = [f for f in os.listdir(Train_dir_in) if 'codefw.npy' in f and 'host' in f and str(cutoff_len)+'_' in f][0]
tr_host_data_r_filename = [f for f in os.listdir(Train_dir_in) if 'codebw.npy' in f and 'host' in f and str(cutoff_len)+'_' in f][0]
print("data for training from -> " + tr_host_data_filename + "  and  " + tr_host_data_r_filename)
tr_host_data = np.load(os.path.join(Train_dir_in,tr_host_data_filename))
tr_host_data_r = np.load(os.path.join(Train_dir_in,tr_host_data_r_filename))

val_host_data_filename = [f for f in os.listdir(Valid_dir_in) if 'codefw.npy' in f and 'host' in f and str(cutoff_len)+'_' in f][0]
val_host_data_r_filename = [f for f in os.listdir(Valid_dir_in) if 'codebw.npy' in f and 'host' in f and str(cutoff_len)+'_' in f][0]
print("data for validatinging from -> " + val_host_data_filename + "  and  " + val_host_data_r_filename)
val_host_data = np.load(os.path.join(Valid_dir_in,val_host_data_filename))
val_host_data_r = np.load(os.path.join(Valid_dir_in,val_host_data_r_filename))

# print(tr_host_data.shape)
# print(tr_host_data_r.shape)
# print(val_host_data.shape)
# print(val_host_data_r.shape)
# print(tr_virus_data.shape)
# print(tr_virus_data_r.shape)
# print(val_virus_data.shape)
# print(val_virus_data_r.shape)

print("......\ncombining virus and host, shuffle training set......")

Y_train = np.concatenate((np.repeat(0,tr_host_data.shape[0]), np.repeat(1,tr_virus_data.shape[0])))
X_train = np.concatenate((tr_host_data,tr_virus_data), axis=0)
X_train_r = np.concatenate((tr_host_data_r,tr_virus_data_r), axis=0)

shuffle_index_train = list(range(0,X_train.shape[0]))
np.random.shuffle(shuffle_index_train)
X_train = X_train[np.ix_(shuffle_index_train, range(X_train.shape[1]), range(X_train.shape[2]))]
X_train_r = X_train_r[np.ix_(shuffle_index_train, range(X_train_r.shape[1]), range(X_train_r.shape[2]))]
Y_train = Y_train[shuffle_index_train]
del tr_host_data, tr_host_data_r, tr_virus_data, tr_virus_data_r

print("......\ncombining virus and host, shuffle validating set......")

Y_valid = np.concatenate((np.repeat(0,val_host_data.shape[0]), np.repeat(1,val_virus_data.shape[0])))
X_valid = np.concatenate((val_host_data,val_virus_data), axis=0)
X_valid_r = np.concatenate((val_host_data_r,val_virus_data_r), axis=0)
del val_host_data, val_host_data_r, val_virus_data, val_virus_data_r

print("......\nStarting set up parameters...")
POOL_FACTOR = 1
dropout_cnn = 0.1
dropout_pool = 0.2
dropout_dense = 0.2
learningrate = 0.001
batch_size=int(X_train.shape[0]/(1000*1000/cutoff_len)) ## smaller batch size can reduce memory
pool_len = int((cutoff_len-filter_len+1)/POOL_FACTOR)

model = 'model_siamese_varlen_'+cutoff_lenk+'k_fl'+str(filter_len)+'_fn'+str(filter_num)+'_dn'+str(dense_num)
modelname = os.path.join(Result_dir_out, model + '.h5')
checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelname, verbose=1,save_best_only=True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1)

print("......\nStarting constructing model...")
def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

if os.path.isfile(modelname):
    model = load_model(modelname)
else :
    forward_input = Input(shape=(None, channel_num))
    reverse_input = Input(shape=(None, channel_num))
    hidden_layers = [ 
	    Conv1D(filters = filter_num, kernel_size = filter_len, activation='relu'),
	    GlobalMaxPooling1D(),
        Dropout(dropout_pool),
        Dense(nb_dense, activation='relu'),
        Dropout(dropout_dense),
        Dense(25, activation='relu'),
        Dropout(dropout_dense),
        Dense(1, activation='sigmoid')
	    ]
    forward_output = get_output(forward_input, hidden_layers)     
    reverse_output = get_output(reverse_input, hidden_layers)
    output = Average()([forward_output, reverse_output])
    model = Model(inputs=[forward_input, reverse_input], outputs=output)
    model.compile(Adam(lr=learningrate), 'binary_crossentropy', metrics=['accuracy'])

print("......\nFitting model......")
print(cutoff_lenk+'_fl'+str(filter_len)+'_fn'+str(filter_num)+'_dn'+str(dense_num)+'_ep'+str(epoch))

if (is_train == 'Y'):
	model.fit(x = [X_train, X_train_r], y = Y_train, \
			batch_size=batch_size, epochs=epoch, verbose=2, \
			validation_data=([X_valid, X_valid_r], Y_valid), \
			callbacks=[checkpointer, earlystopper])

print("......\nStarting predict data......")
print("......\nPredict training set")
Y_train_pred = model.predict([X_train,X_train_r],batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y_train,Y_train_pred)
print('The accuracy is ' + str(auc))

print("Predict validating set")
Y_valid_pred = model.predict([X_valid,X_valid_r],batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y_valid,Y_valid_pred)
print('The accuracy is ' + str(auc))