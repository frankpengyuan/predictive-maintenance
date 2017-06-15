import numpy as np
import tensorflow as tf
from basic_lstm import Data_dim_config, Basic_LSTM

class DeeperCNN_LSTM(Basic_LSTM):
	"""docstring for LSTM_CNN"""
	def __init__(self, 
		num_units = 200,
		learning_rate = 0.001,
		dropout_rate = 0.0,
		config = Data_dim_config(),
		reg_constant = 0.01,
		debug = False):
	
		Basic_LSTM.__init__(self,
			num_units = num_units,
			learning_rate = learning_rate,
			dropout_rate = dropout_rate,
			config = config,
			reg_constant = reg_constant,
			debug = debug)


	"""
	sub graph for placeholders, add placeholders as you need
	"""
	def build_graph_placeholder(self):
		Basic_LSTM.build_graph_placeholder(self)

	"""
	TODO: major part, build to core graph for LSTM_CNN

	you have self.x, self.y, self.keep_prob, and all other placeholders previously defined

	you need to build the sub graph of LSTM_CNN, which takes self.x and return score/logits
	that used for prediction and loss calculation. Store that value in self.logits

	self.x dim: N*L*D, where
		N: days per batch, L: samples per day, D: sensor per sample
	self.y dim: N
	self.logits dim: N*2, i.e. days per batch * scores per class

	"""
	def build_graph_core(self):
		# self.x N*L*D
		# core CNN part
		conv1 = tf.layers.conv1d(
			inputs=self.x,
			filters=100,
			kernel_size=6,
			strides=1,
			padding='valid',
			activation=tf.nn.relu,
			use_bias=True,
			kernel_regularizer=tf.nn.l2_loss
		)

		# pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=1)

		conv2 = tf.layers.conv1d(
			inputs=conv1,
			filters=150,
			kernel_size=10,
			strides=1,
			padding='valid',
			activation=tf.nn.relu,
			use_bias=True,
			kernel_regularizer=tf.nn.l2_loss
		)

		conv3 = tf.layers.conv1d(
			inputs=conv2,
			filters=200,
			kernel_size=20,
			strides=1,
			padding='valid',
			activation=tf.nn.relu,
			use_bias=True,
			kernel_regularizer=tf.nn.l2_loss
		)

		# core LSTM part
		# lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias = 1.0, initializer=tf.random_normal_initializer())
		lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias = 1.0, state_is_tuple=True)
		cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.keep_prob)
		
		score, _ = tf.nn.dynamic_rnn(cell, conv3, dtype=tf.float32) # N*L*h

		self.score = tf.reshape(score[:, -1, :], [-1, self.num_units]) # N*h
		
		self.logits = tf.layers.dense(inputs=self.score, units=2) # N*2

	"""
	other part of the graph, in case you want change it. 
	"""
	def build_graph_other(self):
		Basic_LSTM.build_graph_other(self)
		