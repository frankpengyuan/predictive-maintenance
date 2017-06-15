# Data-driven Predictive Maintenance

## This is a cleaned up version for public view. ##

Peng Yuan pengy@stanford.edu

Rao Zhang zhangrao@stanford.edu

Pan Hu panhu@stanford.edu

## Files and description

code/pre_proc_v7_stable.py is the one we use to preprocess data into .npz files.

code/extract_feature.py for feature extraction for SVMs.

code/logistic_tf.py is Linear SVM with TensorFlow.

code/linear_tf.ipynb is the notebook for Linear SVM.

code/svm_test.ipynb is the notebook for RBF kernel SVM.

code/lstm_cnn/train.py is the main file to train different CNN+LSTM models.

code/lstm_cnn/train.py is the main file to test different CNN+LSTM models.

code/lstm_cnn/util.py contains balancing data, padding, etc.

code/lstm_cnn/all.py lists all the heaters with their number of positive days.

model/DeepCNN_LSTM.model is the best model we currently have, which achieved 0.7549 AUC.

## data after preprocessing should be in the following format:

Each heater is a .npz file, it stores each day's data as a np.array. For each numpy array, it should be a 21\*8640 matrix where 1st column is the timestamp, second column is the label of that day (should be all 0 or all 1), and the rest 19 column are sensor readings.
