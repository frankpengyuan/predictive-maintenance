import numpy as np
import tensorflow as tf
from basic_lstm import Data_dim_config, Basic_LSTM

class LSTM_CNN(Basic_LSTM):
	"""docstring for LSTM_CNN"""
	def __init__(self, 
		num_units = 40,
		learning_rate = 0.001,
		dropout_rate = 0.0,
		config = Data_dim_config(),
		cnn_args = None,
		reg_constant = 0.01,
		debug = False):
	
		Basic_LSTM.__init__(self,
			num_units = num_units,
			learning_rate = learning_rate,
			dropout_rate = dropout_rate,
			config = config,
			reg_constant = reg_constant,
			debug = debug)

		self.cnn_args = cnn_args



	"""
	sub graph for placeholders, add placeholders as you need
	"""
	def build_graph_placeholder(self):
		Basic_LSTM.build_graph_placeholder(self)
		# self.x = tf.placeholder(tf.float32, [None, None, self.config.D], name='x') # N*L*D
		# self.y = tf.placeholder(tf.int32, [None], name='y')	# N
		# self.keep_prob = tf.placeholder("float", name='keep_prob')


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
		# core CNN part
		filter_size = self.cnn_args['filter_size']
		filter_num = self.cnn_args['filter_num']
		sensor_num = self.config.D
		conv1_w = tf.get_variable("conv1_w", shape=[filter_size, sensor_num, filter_num])  # filter_size * in_channel * out_channel
		conv1_b = tf.get_variable("conv1_b", shape=[filter_num]) # out_dim
		
		self.cv1 = tf.nn.conv1d(self.x, conv1_w, stride = 1, padding = 'SAME') + conv1_b
		# h1 = tf.nn.relu(self.cv1)
		
		# core LSTM part
		# lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias = 1.0, initializer=tf.random_normal_initializer())
		lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias = 1.0, state_is_tuple=False)
		cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.keep_prob)
		
		if self.debug and self.__class__ == LSTM_CNN:
			with tf.variable_scope("lstm_cnn_core") as scope:
				init_state = None

				# for l in range(8):
				# 	score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, l*1000:(l+1)*1000, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				# 	self.states.append(init_state)
				# 	scope.reuse_variables()
				# score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, 8000:, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, :7000, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				scope.reuse_variables()
				score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, 7000:8000, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, 8000:8500, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, 8500:8600, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				score, init_state = tf.nn.dynamic_rnn(cell, self.cv1[:, 8600:, :], initial_state=init_state, dtype=tf.float32) # N*L*h

				self.states.append(init_state)
		else:
			score, _ = tf.nn.dynamic_rnn(cell, self.cv1, dtype=tf.float32) # N*L*h

		self.score = tf.reshape(score[:, -1, :], [-1, 1, self.num_units]) # N*1*h
		
		softmax_w = tf.get_variable("softmax_w", [self.num_units, 2], initializer=tf.random_normal_initializer([self.num_units, 2])) # h*2
		softmax_b = tf.get_variable("softmax_b", [2])		  # 2
		
		# perform xw+b on batch, use scan to slice and calc
		logits = tf.scan(lambda a, x: tf.nn.xw_plus_b(x, softmax_w, softmax_b), self.score, np.array([[0.0, 0.0]], dtype=np.float32))					  # N*1*2
		self.logits = tf.reshape(logits, [-1, 2])			  # N*2


	"""
	other part of the graph, in case you want change it. 
	"""
	def build_graph_other(self):
		Basic_LSTM.build_graph_other(self)
		if self.debug and self.__class__ == LSTM_CNN:
			# remove the last grad, which is None
			self.score_grad = tf.gradients(self.loss, self.score)
			self.cv1_grad = tf.gradients(self.loss, self.cv1)

			for s in self.states:
				self.states_grad.append(tf.gradients(self.loss, s)[0])
			# remove the last grad, which is None
			self.states_grad = self.states_grad[:len(self.states_grad)-1]

			# print(self.cv1, self.cv1_grad)


	def fit(self, batch_x, batch_y):
		"""
		batch_x: N*L*D used to train
		batch_y: N labels

		return: (num iter, loss)
		"""
		if self.debug and self.__class__ == LSTM_CNN:
			outputs = self.sess.run([self.loss, self.train_ops, self.score, self.cv1_grad, self.score_grad] + self.states + self.states_grad, 
				feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.keep_prob_val,
				self.batch_size: batch_x.shape[0]})
			# print("iteration", self.it, "loss", outputs[0], "self.cv1", outputs[3], "self.score", outputs[2], "self.score_grad", outputs[4])
			print("iteration", self.it, "loss", outputs[0], "self.cv1_grad", outputs[3])
			print("states_grad", outputs[5+len(self.states):])
		else:
			outputs = self.sess.run([self.loss, self.train_ops, self.score], 
				feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.keep_prob_val,
				self.batch_size: batch_x.shape[0]})
		self.it += 1
		# print(outputs[2])
		return (self.it, outputs[0])	# (num iter, loss)