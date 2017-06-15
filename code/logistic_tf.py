import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LinearClassifier(object):

  def __init__(self, feature_dim, vscope = "LinearClassifier", loss_func = "logistic",
    learning_rate=0.1, reg=1e-5):
    self.vscope = vscope
    self.loss_func = loss_func
    self.learning_rate = learning_rate
    self.reg = reg
    self.feature_dim = feature_dim + 1 # including the bias dim
    self.sess = None

  def build_graph(self, X=None, Y=None):
    if X is not None and Y is not None:
      self.x = tf.constant(X, dtype=tf.float32)
      self.y_ = tf.constant(Y, dtype=tf.float32)
    else:
      self.x = tf.placeholder(tf.float32, [None, self.feature_dim])
      if self.loss_func == "CE":
        self.y_ = tf.placeholder(tf.float32, [None, 2])
      elif self.loss_func in ["logistic", "hinge"]:
        self.y_ = tf.placeholder(tf.float32, [None, 1])

    if self.loss_func == "CE":
      self.W = tf.get_variable("W", [self.feature_dim, 2], initializer=tf.constant_initializer(0.0))
      self.b = tf.get_variable("b", [1, 2], initializer=tf.constant_initializer(0.0))
    elif self.loss_func in ["logistic", "hinge"] :
      self.W = tf.get_variable("W", [self.feature_dim, 1], initializer=tf.random_normal_initializer([self.feature_dim]))
      self.b = tf.get_variable("b", [1, 1], initializer=tf.constant_initializer(0.0))

    self.score = tf.matmul(self.x, self.W) + self.b

    # Define loss and optimizer
    if self.loss_func == "CE":
      data_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.score))
    elif self.loss_func == "logistic":
      data_loss = tf.reduce_mean(tf.log(tf.exp(-tf.multiply(self.y_, self.score)) + 1))
    elif self.loss_func == "hinge":
      factor = tf.where(self.y_ > 0, tf.ones_like(self.y_) * 5000, tf.ones_like(self.y_))
      data_loss = tf.reduce_mean(tf.maximum(0.0, -tf.multiply(self.y_, self.score) + 1))

    self.reg_loss = tf.norm(tf.cast(self.W, tf.float64), ord=2) * self.reg
    self.loss = tf.cast(data_loss, tf.float64) + self.reg_loss
    self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

  def build_eval_graph(self):
    self.x_eval = tf.placeholder(tf.float32, [None, self.feature_dim])
    if self.loss_func == "CE":
      self.y_eval = tf.placeholder(tf.int32, [None, 2])
    elif self.loss_func in ["logistic", "hinge"]:
      self.y_eval = tf.placeholder(tf.int32, [None, 1])

    self.score_eval = tf.matmul(self.x_eval, self.W) + self.b

    # evaluate
    if self.loss_func == "CE":
      correct_prediction = tf.equal(tf.argmax(self.score_eval, 1), tf.argmax(self.y_eval, 1))
    elif self.loss_func == "logistic":
      correct_prediction = tf.equal((1.0 / (tf.exp(-self.score_eval) + 1)) > 0.5, tf.equal(self.y_eval, 1))
    elif self.loss_func == "hinge":
      correct_prediction = tf.equal(self.score_eval > 0, tf.equal(self.y_eval, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # predict
    if self.loss_func == "CE":
      self.pred_y = tf.argmax(self.score_eval, 1)
      self.pred_y_prob = tf.nn.softmax(self.score_eval)[:,1]
    elif self.loss_func == "logistic":
      # y = 1, i.e prob of being class 1
      self.pred_y_prob = (1.0 / (tf.exp(-self.score_eval) + 1))
      self.pred_y = self.pred_y_prob >= 0.5
    elif self.loss_func == "hinge":
      self.pred_y = self.score_eval > 0
      # currently not support prob predict with hinge loss
      self.pred_y_prob = self.score_eval > 0


  def fit(self, X_raw, y_raw, num_iters=1000, batch_size=0, verbose=False, verbose_iter=100, tol=1e-5):
    y_raw = y_raw.astype(np.int32)
    self.X_mean = np.mean(X_raw, axis=0)
    X = X_raw - self.X_mean
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    if self.loss_func == "CE":
      y = np.zeros(shape=(y_raw.shape[0], 2))
      y[range(y_raw.shape[0]), y_raw] = 1
    elif self.loss_func in ["logistic", "hinge"]:
      # convert to 1/-1 label
      y = np.reshape(y_raw, (-1, 1)) * 2 - 1

    loss_history = []
    reg_loss_history = []

    with tf.variable_scope(self.vscope) as scope:
      if batch_size == 0:
        self.build_graph(X=X, Y=y)
      else:
        self.build_graph()
      self.build_eval_graph()

    self.sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    prev_loss = None
    for it in range(num_iters):
      if batch_size > 0:
        indices = np.random.choice(X.shape[0], batch_size, False)
        X_batch = X[indices,:]
        y_batch = y[indices,:]
        outputs = self.sess.run([self.loss, self.train_step, self.reg_loss], feed_dict={self.x: X_batch, self.y_: y_batch})
      else:
        outputs = self.sess.run([self.loss, self.train_step, self.reg_loss])

      loss_history.append(outputs[0])
      reg_loss_history.append(outputs[2])

      if prev_loss and abs(prev_loss-outputs[0]) < tol:
        print('converge at iteration %d / %d: loss %f' % (it, num_iters, outputs[0]))
        break
      else:
        prev_loss = outputs[0]

      if verbose and it % verbose_iter == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, outputs[0]))

    return loss_history
    #return reg_loss_history

  def predict(self, X_raw, prob=False):
    X = X_raw - self.X_mean
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    if prob:
      outputs = self.sess.run(self.pred_y_prob, feed_dict={self.x_eval: X})
    else:
      outputs = self.sess.run(self.pred_y, feed_dict={self.x_eval: X})

    if self.loss_func == "CE":
      return outputs
    elif self.loss_func in ["logistic", "hinge"]:
      return outputs.T.flatten().astype(np.float32)

  def evaluate(self, X_raw, y_raw):
    # Test trained model
    y_raw = y_raw.astype(np.int32)
    X = X_raw - self.X_mean
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    if self.loss_func == "CE":
      y = np.zeros(shape=(y_raw.shape[0], 2))
      y[range(y_raw.shape[0]), y_raw] = 1
    elif self.loss_func in ["logistic", "hinge"]:
      # convert to 1/-1 label
      y = np.reshape(y_raw, (-1, 1)) * 2 - 1

    outputs = self.sess.run(self.accuracy, feed_dict={self.x_eval: X, self.y_eval: y})
    return outputs

  def reset(self):
    self.sess.close()
    tf.reset_default_graph()

if __name__ == '__main__':
  classifier = LinearClassifier(feature_dim=3325, loss_func="CE")

  train_data_raw = []
  train_dir = "E:\\cs341-project\\iotdata\\npz_feats_all\\train"
  for one_file in os.listdir(train_dir):
    if one_file.endswith(".feats.npz"):
      train_data_raw.append(np.load(os.path.join(train_dir, one_file))["arr_0"])

  train_data = np.concatenate(train_data_raw)
  X = train_data[:, 1:]
  Y = train_data[:, 0]

  test_data_raw = []
  test_dir = "E:\\cs341-project\\iotdata\\npz_feats_all\\test"
  for one_file in os.listdir(test_dir):
    if one_file.endswith(".feats.npz"):
      test_data_raw.append(np.load(os.path.join(test_dir, one_file))["arr_0"])

  test_data = np.concatenate(train_data_raw)
  X_test = test_data[:, 1:]
  Y_test = test_data[:, 0]

  loss_hist = classifier.fit(X, Y, batch_size=0, num_iters=100, verbose_iter=10, verbose=True)
  print(classifier.evaluate(X_test, Y_test))