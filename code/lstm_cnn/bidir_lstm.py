import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from basic_lstm import Data_dim_config, Basic_LSTM


class BiLSTM(Basic_LSTM):
	"""docstring for BiLSTM"""
	def __init__(self, 
		num_units = 40,
		learning_rate = 0.001,
		dropout_rate = 0.0,
		batch_size = None,
		config = Data_dim_config(),
		your_args=None,
		debug = False):

		Basic_LSTM.__init__(self,
			num_units = num_units,
			learning_rate = learning_rate,
			dropout_rate = dropout_rate,
			#batch_size = batch_size,
			config = config,
			debug = debug)

		#self.your_args = your_args
		self.keep_prob_val = 1.0 - dropout_rate
	


	"""
	sub graph for placeholders, add placeholders as you need
	"""
	def build_graph_placeholder(self):
		Basic_LSTM.build_graph_placeholder(self)

		#your_placeholder = ... (if any)   

	"""
	TODO: major part, build to core graph for BiLSTM

	you have self.x, self.y, self.keep_prob, and all other placeholders previously defined

	you need to build the sub graph of BiLSTM, which takes self.x and return score/logits
	that used for prediction and loss calculation. Store that value in self.logits

	self.x dim: N*L*D, where
		N: days per batch, L: samples per day, D: sensor per sample
	self.y dim: N
	self.logits dim: N*2, i.e. days per batch * scores per class

	"""
	def build_graph_core(self):
		with tf.name_scope("BiLSTM"):
			with tf.variable_scope('forward'):
				# Forward path        
				fw_lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias=1.0, state_is_tuple=False)
				fw_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=self.keep_prob)
				fw_score, _ = tf.nn.dynamic_rnn(fw_cell, self.x, dtype=tf.float32) # N*L*h   
			with tf.variable_scope('backward'):
				# Backward path
				bw_lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias=1.0, state_is_tuple=False)
				bw_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=self.keep_prob)
				bw_score, _ = tf.nn.dynamic_rnn(bw_cell, self.x_r, dtype=tf.float32) # N*L*h
			self.score = tf.concat([fw_score, bw_score], -1)
			self.score = tf.reshape(self.score[:, -1, :], [-1, 1, 2*self.num_units]) # N*1*2h
		# get logits from score
		softmax_w = tf.get_variable("softmax_w", [2*self.num_units, 2], initializer=tf.random_normal_initializer([self.num_units, 2])) # h*2
		softmax_b = tf.get_variable("softmax_b", [2])	# 2
		# perform xw+b on batch, use scan to slice and calc
		logits = tf.scan(lambda a, x: tf.nn.xw_plus_b(x, softmax_w, softmax_b), self.score, np.array([[0.0, 0.0]], dtype=np.float32)) # N*1*2
		self.logits = tf.reshape(logits, [-1, 2]) # N*2



	"""
	other part of the graph, in case you want change it. 
	"""
	def build_graph_other(self):
		Basic_LSTM.build_graph_other(self)
