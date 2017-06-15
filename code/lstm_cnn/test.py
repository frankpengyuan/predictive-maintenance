import os
import numpy as np
from util import Heater_Data, gen_sim_data, Solver
from basic_lstm import Basic_LSTM
from lstm_cnn import LSTM_CNN
from bidir_lstm import BiLSTM
from lstm_cnn_deep import DeepCNN_LSTM
from lstm_cnn_deeper import DeeperCNN_LSTM

data_dir = "/home/frankpengyuan/cs341/npz_data_all/"
model_dir = "/home/frankpengyuan/cs341/model/"
code_dir = os.getcwd()

# previously train.txt and test.txt are in data_dir, comment next line if they are in code_dir, uncomment if they are in data_dir
# os.chdir(data_dir)

train_f = []
test_f = []
with open("train_large.txt") as train_fnames:
    for line in train_fnames:
        train_f.append(line.split(" ")[0])
with open("test_large.txt") as test_fnames:
    for line in test_fnames:
        test_f.append(line.split(" ")[0])

# setup model
lstm_model = Basic_LSTM(num_units = 80, learning_rate = 0.001, dropout_rate = 0.5, debug=False)
lstm_cnn_model = LSTM_CNN(num_units = 100, learning_rate = 0.001, dropout_rate = 0.5, cnn_args = {'filter_size': 30, 'filter_num': 48}, debug=False)
bilstm_model = BiLSTM(learning_rate = 0.001, dropout_rate = 0.5, debug=False)
deep_cnn_lstm_model = DeepCNN_LSTM(dropout_rate = 0.0, reg_constant = 0.01)
deeper_cnn_lstm_model = DeeperCNN_LSTM(dropout_rate = 0.0, reg_constant = 0.01)

# select which one to use
model = deep_cnn_lstm_model

# data config
D_config = {"batch_size":90, "shuffle":False, "balance":False, "padding":False, "free_mem":True}
os.chdir(data_dir)

train_cfg = {
    "model":model,
    "max_iter":1000,
    "data_dir":data_dir,
    "fnames":train_f,
    "D_config":D_config,
    "model_save_dir":model_dir,
    "save_every_iter":200,
    "full_batch":True
}

# model.initial()
model.initial(os.path.join(model_dir, "DeepCNN_LSTM-1000.model"))
m_solver = Solver(num_file_in_mem=5, num_threads=None) # num_threads=None means to use all cores
# m_solver(**train_cfg)
m_solver.test(model=model, data_dir=data_dir, fnames=test_f, D_config=D_config)
model.finish()
