import numpy as np
import tensorflow as tf
import os

class Data_dim_config(object):
	"""docstring for Data_dim_config"""
	def __init__(self):
		self.D = 19
		self.L = 8640
		
class Basic_LSTM(object):
	"""docstring for Basic_LSTM"""
	def __init__(self, 
		num_units = 40,
		learning_rate = 0.001,
		dropout_rate = 0.0,
		config = Data_dim_config(),
		reg_constant = 0.01,
		debug = False
		):

		self.num_units = num_units
		self.learning_rate = learning_rate
		self.config = config
		self.keep_prob_val = 1.0 - dropout_rate
		self.reg_constant = reg_constant

		self.debug = debug
		self.states = []
		self.states_grad = []

	def initial(self, fname=None):
		"""
		build graph and initlial variables
		or load variables from saved model at fname
		fname should be full path
		"""
		self.it = 0 # number of iterations

		self.build_graph_placeholder()
		self.build_graph_core()
		self.build_graph_other()

		self.saver = tf.train.Saver()

		self.sess = tf.InteractiveSession() # tf session
		if fname is None:
			tf.global_variables_initializer().run()
		else:
			self.saver.restore(self.sess, fname)
			print("model restored.")

	def save_model(self, fname):
		save_path = self.saver.save(self.sess, fname)
		print("Model saved in file: %s" % save_path)

	def build_graph_placeholder(self):
		self.x = tf.placeholder(tf.float32, [None, None, self.config.D], name='x') # N*L*D
		# self.x = tf.placeholder(tf.float32, [None, self.config.L, self.config.D], name='x') # N*L*D
		self.y = tf.placeholder(tf.int32, [None], name='y')	# N
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		self.batch_size = tf.placeholder(tf.float32, name='batch_size')

	def build_graph_core(self):
		# core LSTM part
		# lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=0.5, state_is_tuple=False)
		# lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias=0.98, state_is_tuple=False, initializer=tf.random_normal_initializer())
		
		# TODO: if bug fixed, use state_is_tuple=False to speed up
		# TOCO: try initializer=tf.random_normal_initializer() to see the difference
		lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units, forget_bias=1.0, state_is_tuple=False)
		cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

		if self.debug and self.__class__ == Basic_LSTM:
			with tf.variable_scope("lstm_core") as scope:
				init_state = None

				# for l in range(8):
				# 	score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, l*1000:(l+1)*1000, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				# 	self.states.append(init_state)
				# 	scope.reuse_variables()
				# score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, 8000:, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, :7000, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				scope.reuse_variables()
				score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, 7000:8000, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, 8000:8500, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, 8500:8600, :], initial_state=init_state, dtype=tf.float32) # N*L*h
				self.states.append(init_state)
				score, init_state = tf.nn.dynamic_rnn(cell, self.x[:, 8600:, :], initial_state=init_state, dtype=tf.float32) # N*L*h

				self.states.append(init_state)
		else:
			score, _ = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32) # N*L*h

		self.score = tf.reshape(score[:, -1, :], [-1, 1, self.num_units]) # N*1*h

		# get logits from score
		softmax_w = tf.get_variable("softmax_w", [self.num_units, 2], initializer=tf.random_normal_initializer([self.num_units, 2])) # h*2
		softmax_b = tf.get_variable("softmax_b", [2])	# 2
		# perform xw+b on batch, use scan to slice and calc
		logits = tf.scan(lambda a, x: tf.nn.xw_plus_b(x, softmax_w, softmax_b), self.score, np.array([[0.0, 0.0]], dtype=np.float32)) # N*1*2
		self.logits = tf.reshape(logits, [-1, 2]) # N*2

	def build_graph_other(self):
		# get loss from logits, using softmax CE
		data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='CE')
		reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		self.loss = tf.reduce_sum(data_loss) / self.batch_size \
			+ self.reg_constant * sum(reg_loss)

		if self.debug and self.__class__ == Basic_LSTM:
			for s in self.states:
				self.states_grad.append(tf.gradients(self.loss, s)[0])
			# remove the last grad, which is None
			self.states_grad = self.states_grad[:len(self.states_grad)-1]
		
		# get optimizer
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_ops = self.optimizer.minimize(self.loss)

		# for prediction
		self.prediction_prob = tf.nn.softmax(self.logits) # N*2
		self.prediction = tf.argmax(self.prediction_prob, axis=1) # N

	def fit(self, batch_x, batch_y):
		"""
		batch_x: N*L*D used to train
		batch_y: N labels

		return: (num iter, loss)
		"""
		if self.debug and self.__class__ == Basic_LSTM:
			outputs = self.sess.run([self.loss, self.train_ops, self.score] + self.states + self.states_grad, 
				feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.keep_prob_val,
				self.batch_size: batch_x.shape[0]})
			print("iteration", self.it, "loss", outputs[0])
			# print("states", outputs[3:3+len(self.states)])
			print("states_grad", outputs[3+len(self.states):])
		else:
			outputs = self.sess.run([self.loss, self.train_ops, self.score], 
				feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.keep_prob_val,
				self.batch_size: batch_x.shape[0]})
		self.it += 1
		# print(outputs[2])
		return (self.it, outputs[0])	# (num iter, loss)

	def predict(self, test_x, test_y=None):
		"""
		test_x: N*L*D used to predict
		test_y: correct labels, used to calc loss if provided

		return: (loss, N predictions)
		"""
		if test_y is not None:
			outputs = self.sess.run([self.loss, self.prediction], 
				feed_dict={self.x: test_x, self.y: test_y, self.keep_prob: 1.0,
				self.batch_size: test_x.shape[0]})
			return outputs # (loss, 0/1 prediction)
		else:
			outputs = self.sess.run([self.prediction], 
				feed_dict={self.x: test_x, self.keep_prob: 1.0, self.batch_size: test_x.shape[0]})
			return -1, outputs # (loss, 0/1 prediction)

	def predict_prob(self, test_x, test_y=None):
		"""
		test_x: N*L*D used to predict
		test_y: correct labels, used to calc loss if provided

		return: (loss, N*2 prob tensor)
		"""
		if test_y is not None:
			outputs = self.sess.run([self.loss, self.prediction_prob, self.score], 
				feed_dict={self.x: test_x, self.y: test_y, self.keep_prob: 1.0,
				self.batch_size: test_x.shape[0]})
			# print(outputs[2])
			return outputs[0:2] # (loss, N*2 prob tensor)
		else:
			outputs = self.sess.run([self.prediction_prob], 
				feed_dict={self.x: test_x, self.keep_prob: 1.0, self.batch_size: test_x.shape[0]})
			return -1, outputs # (loss, N*2 prob tensor)

	def get_raw_logits(self, test_x, test_y=None):
		"""
		test_x: N*L*D used to predict
		test_y: correct labels, used to calc loss if provided

		return: (loss, N*2 logits tensor)
		"""
		if test_y is not None:
			outputs = self.sess.run([self.loss, self.logits], 
				feed_dict={self.x: test_x, self.y: test_y, self.keep_prob: 1.0,
				self.batch_size: test_x.shape[0]})
			# print(outputs[2])
			return outputs[0:2] # (loss, N*2 prob tensor)
		else:
			outputs = self.sess.run([self.logits], 
				feed_dict={self.x: test_x, self.keep_prob: 1.0, self.batch_size: test_x.shape[0]})
			return -1, outputs # (loss, N*2 prob tensor)

	def finish(self):
		"""
		end session, reset the graph, and free mem
		"""
		self.sess.close()
		tf.reset_default_graph()
