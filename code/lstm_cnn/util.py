import numpy as np
import random
from functools import reduce
from multiprocessing.pool import Pool, AsyncResult
import time
import collections
import os
from sklearn.metrics import roc_auc_score

class dataError(Exception):
	pass
		
class Heater_Data():
	"""docstring for heater_data"""
	def __init__(self, fname):
		self.fname = fname
		self.configed = False
		self.loaded_mem = False

	def config(self, batch_size, shuffle=False, balance=False, padding=False, normalize=True, free_mem=True):
		"""
		batch_size: int, number of days per batch
		shuffle: bool, indicating doing shuffle or not
		balance: 
			dict{0:w1, 1:w2, min_n:10} - use balanced data with number of neg/pos sampels 
			according to w1/w2, occurs before shuffle, minimal number of data selected in min_n
			False - if not use balancing
		padding: bool, padding 0s if remaining samples less than batch_size
		normalize: bool, normalize data on each sensor
		free_mem: 
			True - auto free mem after all data used
			False - reset read_pos to beginning position of the file

		return: None
		"""
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.balance = balance
		self.padding = padding
		self.normalize = normalize
		self.free_mem = free_mem
		if self.balance and "min_n" not in self.balance:
			self.balance["min_n"] = self.batch_size
		self.configed = True
		self.min_max_val = np.array([[-1.0, 1.0],
									 [0.0, 100.0],  # 1. Actual_Power
									 [0.0, 585638.0],  # 2. Number_of_burner_starts
									 [0.0, 1.0],  # 3. Operating_status:_Central_heating_active
									 [0.0, 1.0],  # 4. Operating_status:_Hot_water_active
									 [0.0, 1.0],  # 5. Operating_status:_Flame
									 [0.0, 1.0],  # 6. Relay_status:_Gasvalve
									 [0.0, 1.0],  # 7. Relay_status:_Fan
									 [0.0, 1.0],  # 8. Relay_status:_Ignition
									 [0.0, 1.0],  # 9. Relay_status:_CH_pump
									 [0.0, 1.0],  # 10.Relay_status:_internal_3-way-valve
									 [-1.0, 1.0],  # 11.Relay_status:_HW_circulation_pump
									 [0.0, 99.0],  # 12.Supply_temperature_(primary_flow_temperature)_setpoint
									 [0.0, 99.0],  # 13.Supply_temperature_(primary_flow_temperature) !!!!!! -3276.8
									 [0.0, 100.0],  # 14.CH_pump_modulation
									 [0.0, 99.0],  # 15.Maximum_supply_(primary_flow)_temperature
									 [0.0, 99.0],  # 16.Hot_water_temperature_setpoint
									 [0.0, 99.0],  # 17.Hot_water_outlet_temperature !!!!!!! -3276.8
									 [0.0, 9.0],  # 18.Actual_flow_rate_turbine
									 [0.0, 4750.0]  # 19.Fan_speed
		])
		self.mean_val = np.array([0.5,
								  5.45855861e+00, # 1. Actual_Power
								  7.26133938e+04, # 2. Number_of_burner_starts (target)
								  1.30350094e-01, # 3. Operating_status:_Central_heating_active
								  3.20485423e-02, # 4. Operating_status:_Hot_water_active
								  1.20255331e-01, # 5. Operating_status:_Flame
								  1.24275171e-01, # 6. Relay_status:_Gasvalve
								  1.43160549e-01, # 7. Relay_status:_Fan
								  3.01794562e-03, # 8. Relay_status:_Ignition
								  1.98021219e-01, # 9. Relay_status:_CH_pump
								  7.15357540e-01, # 10.Relay_status:_internal_3-way-valve
								  5.82772126e-05, # 11.Relay_status:_HW_circulation_pump
								  55.0, # 12.Supply_temperature_(primary_flow_temperature)_setpoint (target)
								  # 6.25251225e+01, # 12.Supply_temperature_(primary_flow_temperature)_setpoint (target)
								  55.0, # 13.Supply_temperature_(primary_flow_temperature) !!!!!! -3276.8 (target)
								  # 5.03859511e+01, # 13.Supply_temperature_(primary_flow_temperature) !!!!!! -3276.8 (target)
								  1.68504783e+01, # 14.CH_pump_modulation
								  6.70302895e+01, # 15.Maximum_supply_(primary_flow)_temperature (target)
								  45.0, # 16.Hot_water_temperature_setpoint (target)
								  # 5.25431720e+01, # 16.Hot_water_temperature_setpoint (target)
								  45.0, # 17.Hot_water_outlet_temperature !!!!!!! -3276.8 (target)
								  # 3.98978739e+01, # 17.Hot_water_outlet_temperature !!!!!!! -3276.8 (target)
								  2.86698779e-02, # 18.Actual_flow_rate_turbine
								  2.15416539e+01])# 19.Fan_speed
		self.std_val = np.array([1.0,
								 1.60757180e+01, # 1. Actual_Power
								 6.28186053e+04, # 2. Number_of_burner_starts (target)
								 3.36688204e-01, # 3. Operating_status:_Central_heating_active
								 1.76129024e-01, # 4. Operating_status:_Hot_water_active
								 3.25259875e-01, # 5. Operating_status:_Flame
								 3.29895215e-01, # 6. Relay_status:_Gasvalve
								 3.50236500e-01, # 7. Relay_status:_Fan
								 5.48528726e-02, # 8. Relay_status:_Ignition
								 3.98508238e-01, # 9. Relay_status:_CH_pump
								 4.51243981e-01, # 10.Relay_status:_internal_3-way-valve
								 7.63372886e-03, # 11.Relay_status:_HW_circulation_pump
								 # 1.66415933e+01, # 12.Supply_temperature_(primary_flow_temperature)_setpoint (target)
								 15.0, # 12.Supply_temperature_(primary_flow_temperature)_setpoint (target)
								 # 1.27710949e+01, # 13.Supply_temperature_(primary_flow_temperature) !!!!!! -3276.8 (target)
								 15.0, # 13.Supply_temperature_(primary_flow_temperature) !!!!!! -3276.8 (target)
								 3.55657838e+01, # 14.CH_pump_modulation
								 7.91994647e+00, # 15.Maximum_supply_(primary_flow)_temperature (target)
								 # 3.41892390e+00, # 16.Hot_water_temperature_setpoint (target)
								 8.0, # 16.Hot_water_temperature_setpoint (target)
								 # 1.09264908e+01, # 17.Hot_water_outlet_temperature !!!!!!! -3276.8 (target)
								 8.0, # 17.Hot_water_outlet_temperature !!!!!!! -3276.8 (target)
								 4.21267009e-01, # 18.Actual_flow_rate_turbine
								 2.49952302e+02])# 19.Fan_speed

	def load_into_mem(self):
		"""
		read .npz file and load data into memory
		return: None
		"""
		if not self.configed:
			raise dataError("need config before load_into_mem")

		# in each [L, D], Time at col 0, label at col 1, data start at col 2:
		f = np.load(self.fname)

		keys = sorted([key for key in f], key=lambda x: int(x.split("_")[1]))

		# data_frame: N * L * D
		# in each [L, D], label at col 0, data start at col 1:
		# here remove time_stamp at col 0 in f
		self.data_frame = [f[k][:,1:].astype(np.float) for k in keys]

		# some data has L = 17280, converting to 8640
		for x in range(len(self.data_frame)):
			if self.data_frame[x].shape[0] > 10000:
				self.data_frame[x] = self.data_frame[x][::2]

		if self.balance:
			n_pos = reduce(lambda x, y: x+y, map(lambda x: x[0][0], self.data_frame))
			n_neg = int(self.balance[0] / self.balance[1] * n_pos)
			# make sure there are minial min_n data samples
			if n_neg < self.balance["min_n"] - n_pos:
				n_neg = int(self.balance["min_n"] - n_pos)

			pos_data = list(filter(lambda x: x[0][0] == 1, self.data_frame))
			neg_data = []
			if n_neg > 0:
				neg_data = list(filter(lambda x: x[0][0] == 0, self.data_frame))

				# if remaining data less than n_neg, use all of them
				# else
				if len(neg_data) > n_neg:
					# idx = np.random.choice(range(n_neg), n_neg, replace=False)
					# neg_data = np.array(neg_data)[idx]
					# print(len(neg_data), n_neg)
					neg_data = random.sample(neg_data, n_neg)
			self.data_frame = pos_data
			self.data_frame.extend(neg_data)

		if self.shuffle:
			np.random.shuffle(self.data_frame)

		if self.normalize:
			# Done: linear scaler to range [-1, 1]
			# D = 20 not 19
			# npdf = np.array(self.data_frame)
			# ave = np.mean(npdf, axis=(0, 1), dtype=np.float64, keepdims=False)[np.newaxis, :] # 1 * D
			# max_val = (np.amax(npdf - ave, axis=(0, 1), keepdims=False) + 1e-12)[np.newaxis, :] # 1 * D
			# self.data_frame = [(f - ave) / max_val for f in self.data_frame] # N * L * D
			
			# avera = np.mean(self.min_max_val, axis = 1).T 					  # 1 * D
			# scale = 2.0 / (self.min_max_val[:, 1] - self.min_max_val[:, 0]).T # 1 * D
			# self.data_frame = [(f - avera) * scale for f in self.data_frame]
			
			# all_col = np.arange(0, 20)
			# target_col = np.array([2, 12, 13, 15, 16, 17])
			# nontarget_col = np.setxor1d(all_col, target_col)
			# temp_data_frame = []
			# for f in self.data_frame:
			# 	f_mask = f > 0.5				# set mask
			# 	f_mask[:, nontarget_col] = 1	# set nontargeted columns to 1
			# 	ff = ((f - self.mean_val) / (self.std_val * 5.0 + 1e-12 )) * f_mask	# mask the frame
			# 	temp_data_frame.append(ff)
			# self.data_frame = temp_data_frame

			# npdf = np.array(self.data_frame)
			# ave = np.mean(npdf, axis=(0, 1), dtype=np.float64, keepdims=False)[np.newaxis, :] # 1 * D
			# std = np.std(npdf, axis=(0, 1), dtype=np.float64, keepdims=False)[np.newaxis, :] # 1 * D
			# max_val = (np.amax(npdf - ave, axis=(0, 1), keepdims=False) + 1e-12)[np.newaxis, :] # 1 * D
			# self.data_frame = [(f - ave) / (5*std+1e-12) for f in self.data_frame] # N * L * D
			
			npdf = np.array(self.data_frame)
			ave = np.mean(npdf, axis=(0, 1), dtype=np.float64, keepdims=False)[np.newaxis, :] # 1 * D
			ave[0, 12] = (ave[0, 12]+ave[0, 13]) / 2.0
			ave[0, 13] = (ave[0, 12]+ave[0, 13]) / 2.0
			ave[0, 16] = (ave[0, 16]+ave[0, 17]) / 2.0
			ave[0, 17] = (ave[0, 16]+ave[0, 17]) / 2.0
			max_val = (np.amax(npdf - ave, axis=(0, 1), keepdims=False) + 1e-12)[np.newaxis, :] # 1 * D
			max_val[0, 12] = max(max_val[0, 12], max_val[0, 13])
			max_val[0, 13] = max(max_val[0, 12], max_val[0, 13])
			max_val[0, 16] = max(max_val[0, 16], max_val[0, 17])
			max_val[0, 17] = max(max_val[0, 16], max_val[0, 17])
			self.data_frame = [(f - ave) / max_val for f in self.data_frame] # N * L * D
			

		self.loaded_mem = True
		self.read_pos = 0

	def get_batch(self):
		"""
		Get a batch of data, need load_into_mem() first
		return: batched data according to config; return None if all data has been used
		"""
		if not self.loaded_mem:
			raise dataError("need load_into_mem before get_batch")

		if self.data_frame is None or len(self.data_frame) == 0:
			return []
			# return None

		if len(self.data_frame) - self.read_pos <= self.batch_size:
			if self.padding:
				padding_len = self.batch_size - (len(self.data_frame) - self.read_pos)
				padding_arry = [np.zeros_like(self.data_frame[0])] * padding_len
				rtn_data = self.data_frame[self.read_pos:]
				rtn_data.extend(padding_arry)
				if self.free_mem:
					self.data_frame = None	# free memory
				else:
					self.read_pos = 0
				# return np.array(rtn_data)
				return rtn_data

			rtn_data = self.data_frame[self.read_pos:]
			if self.free_mem:
				self.data_frame = None	# free memory
			else:
				self.read_pos = 0
			# return np.array(rtn_data)
			return rtn_data

		rtn_data = self.data_frame[self.read_pos : self.read_pos+self.batch_size]
		self.read_pos += self.batch_size
		#print(self.read_pos, len(self.data_frame), len(rtn_data))

		# return np.array(rtn_data)
		return rtn_data # list of np array, each element is one day's data

class Solver(object):
	"""
	Solver class, wrapper of the model, involving multithreading to load data,
	which significantly improve the training/testing efficiency
	"""
	def __init__(self, num_file_in_mem=10, num_threads=None):
		"""
		num_file_in_mem: int, number of files kept in memory
		num_threads: int, max processes for loading the data;
			if it is None, it will be set to # of cores of the PC
		"""
		self.num_threads = num_threads
		self.num_file_in_mem = num_file_in_mem
		# cur_file_in_mem track current files stored in mem
		self.cur_file_in_mem = 0

		# queue for Heater_Data class returned by each thread/process
		self.DataPool = collections.deque()
		self.full_batch_pool = []

		# for testing and calc AUC
		self.all_y = []
		self.all_pred = []

	def data_dir2target_files(self, data_dir, fnames):
		os.chdir(data_dir)
		all_files = os.listdir(data_dir)
		target_files = []

		for one_file in all_files:
			if one_file.endswith(".npz") and one_file.split(".")[-2] in fnames:
				target_files.append(one_file)

		if len(target_files) == 0:
			print("no files selected")
			return None

		# shuffle heaters before training
		random.shuffle(target_files)
		return target_files

	def train_main(self, model, model_save_dir, save_every_iter, pool, target_files, max_iter, full_batch):
		# full_batch: accumulate and run on full batch
			
		# track time
		tic = time.clock()

		D = self.DataPool.popleft()
		
		if full_batch:
			d = D.get_batch()
		else:
			d = np.array(D.get_batch())

		while len(d) > 0:
			if full_batch:
				self.full_batch_pool.extend(d)
				if len(self.full_batch_pool) >= self.D_config["batch_size"]:
					da = np.array(self.full_batch_pool[:self.D_config["batch_size"]])
					self.full_batch_pool = self.full_batch_pool[self.D_config["batch_size"]:]
				else:
					d = D.get_batch()
					continue
			else:
				da = d

			# da is None after all data in that heater used
			y = (da[:, 0, 0] > 0).astype(np.int32)
			x = da[:, :, 1:]
			# train one iter
			it, loss = model.fit(x, y)
			# track time
			toc = time.clock()
			print("[%.2fs] iter: %d, loss: %.4f," % (toc-tic, it, loss), "input dim:", da.shape)
			# track time
			tic = time.clock()

			# save model if it % save_every_iter == 0
			if model_save_dir is not None and it % save_every_iter == 0:
				model.save_model(os.path.join(model_save_dir, "%s-%d.model" % (self.model_class, it)))
			# stop trainning if reach max_iter
			if it >= max_iter:
				pool.close()
				pool.join()
				self.DataPool.clear()
				return
			# in case of all data in memory, d wont be None, but read_pos will be reset to 0
			if self.num_file_in_mem >= len(target_files):
				if D.read_pos == 0:
					# put D back so that we can train on it again in the future
					self.DataPool.append(D)
					break

			if full_batch:
				d = D.get_batch()
			else:
				d = np.array(D.get_batch())

		if self.num_file_in_mem < len(target_files):
			self.cur_file_in_mem -= 1

	def __call__(self, model, max_iter, data_dir, fnames, D_config, model_save_dir=None, save_every_iter=1000, full_batch=False):
		"""
		model: LSTM/LSTM_CNN/BiLSTM class, model for training, 
			model should be initialized/loaded before passing in
		max_iter: max iterations for training
		data_dir: .npz file dir
		fnames: list of file names for training/testing
		D_config: data loader config
		model_save_dir: string, folder dir where to save all models
		save_every_iter: int, save the model into model_save_dir every save_every_iter iterations
		"""
		# auto save model according to class name
		self.model_class = model.__class__.__name__

		self.D_config = D_config

		target_files = self.data_dir2target_files(data_dir, fnames)
		if target_files is None:
			return None

		if self.num_file_in_mem >= len(target_files):
			# if all heater data can fit in memory
			self.D_config["free_mem"] = False
		else:
			self.D_config["free_mem"] = True

		file_idx = 0
		it = 0

		# process based thread pool
		# API: https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing
		pool = Pool(self.num_threads)
		load_flag = False

		print("[INFO] Notice that we are using multiprocessing to load files, so that child processes won't print out on ipython-notebook, which only monitor the parent process. Please check the terminal for more logging info.")
		self.cur_file_in_mem = 0
		while True:
			if self.num_file_in_mem >= len(target_files):
				# if all data in mem, should wait untill all the data is ready
				# otherwise will run multiple times on data that loaded first 
				if not load_flag:
					for one_file in target_files:
						pool.apply_async(self._newthread_helper, args=(one_file,), \
							callback=self._callback_helper, error_callback=self._error_helper)
					pool.close()
					# wait till all loaded
					pool.join()

					# flag up, then we wont load data again
					load_flag = True
			else:
				one_file = target_files[file_idx]
				if self.cur_file_in_mem < min(self.num_file_in_mem, len(target_files)):
					# start a new process loading one_file
					pool.apply_async(self._newthread_helper, args=(one_file,), \
						callback=self._callback_helper, error_callback=self._error_helper)
					self.cur_file_in_mem += 1
					file_idx = (file_idx + 1) % len(target_files)
					if file_idx == 0:
						# shuffle target_files if we finish one round on all of them
						random.shuffle(target_files)
				else:
					time.sleep(0.001)

			if len(self.DataPool) > 0:
				# some process returned Heater_Data into self.DataPool
				self.train_main(model, model_save_dir, save_every_iter, pool, target_files, max_iter, full_batch)
			else:
				time.sleep(0.001)
		pool.close()
		pool.join()
		# reset DataPool for another training/testing
		self.DataPool.clear()

	def test(self, model, data_dir, fnames, D_config, use_logits=False):
		"""
		model: LSTM/LSTM_CNN/BiLSTM class, model for testing, should be loaded before pass in
		data_dir: .npz file dir
		fnames: list of file names for training/testing
		D_config: data loader config
		"""
		self.D_config = D_config
		self.D_config["free_mem"] = True
		self.use_logits = use_logits

		# reset for testing
		self.all_y = []
		self.all_pred = []

		target_files = self.data_dir2target_files(data_dir, fnames)
		if target_files is None:
			return None

		file_idx = 0

		# indicating that all file started loading
		all_done = False

		pool = Pool(self.num_threads)
		print("[INFO] Notice that we are using multiprocessing to load files, so that child processes won't print out on ipython-notebook, which only monitor the parent process. Please check the terminal for more logging info.")
		self.cur_file_in_mem = 0
		while not all_done:
			one_file = target_files[file_idx]
			if self.cur_file_in_mem < self.num_file_in_mem and not all_done:
				# start a new process for loading test data
				pool.apply_async(self._newthread_helper, args=(one_file,), \
					callback=self._callback_helper, error_callback=self._error_helper)
				self.cur_file_in_mem += 1
				file_idx = file_idx + 1
				if file_idx == len(target_files):
					# all file started loading
					all_done = True
			else:
				time.sleep(0.001)

			if len(self.DataPool) > 0:
				self._batch_test_helper(model)
			else:
				time.sleep(0.001)

		pool.close()
		# wait all child process done, i.e. put all Heater_Data into self.DataPool
		pool.join()
		for _ in range(len(self.DataPool)):
			self._batch_test_helper(model)

		# calc overall accuracy and AUC
		self.all_y_onehot = np.concatenate(self.all_y)
		self.all_y = np.argmax(self.all_y_onehot, axis=1)
		self.all_pred = np.concatenate(self.all_pred)
		pred_y = np.argmax(self.all_pred, axis=1)
		m_auc = roc_auc_score(self.all_y_onehot, self.all_pred)
		print("overall acc: %.4f, overall AUC: %.4f" % (np.mean(pred_y==self.all_y), m_auc))
		# reset self.DataPool for future training/testing
		self.DataPool.clear()
		return m_auc

	def _batch_test_helper(self, model):
		"""
		helper function for testing on one Heater's data
		"""
		D = self.DataPool.popleft()
		d = np.array(D.get_batch())
		while len(d) > 0:
			y = (d[:, 0, 0] > 0).astype(np.int32)
			x = d[:, :, 1:]
			one_hot_y = np.zeros((y.size, 2))
			one_hot_y[np.arange(y.size), y] = 1

			if self.use_logits:
					loss, pred = model.get_raw_logits(x, y)
					# pred is N*2 logits tensor
			else:
					loss, pred = model.predict_prob(x, y)
					# pred is N*2 prob tensor
			pred_y = np.argmax(pred, axis=1)


			try:
					print("batch shape:", x.shape, "loss: %.4f, acc: %.4f, AUC: %.4f" \
							% (loss, np.mean(pred_y==y), roc_auc_score(one_hot_y, pred)))
			except Exception:
					print("batch shape:", x.shape, "loss: %.4f, acc: %.4f, AUC: N/A" \
							% (loss, np.mean(pred_y==y)))
			# store y and pred to calc overall acc/AUC
			self.all_y.append(one_hot_y)
			self.all_pred.append(pred)
			d = np.array(D.get_batch())
		# keep track how many files in mem
		self.cur_file_in_mem -= 1

	def _newthread_helper(self, fname):
		"""
		helper function for loading a file
		"""
		print ("start loading %s" % (fname,))
		tic = time.clock()

		D = Heater_Data(fname)
		D.config(**self.D_config)
		D.load_into_mem()

		toc = time.clock()
		print ("[%ds] finished loading %s" % (toc - tic, fname))
		return D

	def _callback_helper(self, D):
		# callback function after one precess finishes, store Heater_Data D into DataPool
		self.DataPool.append(D)
	
	def _error_helper(self, err):
		# error handler for child process
		print(err)

def gen_sim_data():
	x = np.array(list(np.array([list(range(20))] * 19).T) * 42)
	x_ = np.array(list(np.array([list(range(0, -20, -1))] * 19).T) * 42) + 20
	X = np.stack([x] * 10) # 10 * (20*42) * 19
	X = X + np.random.uniform(0, 20, X.shape)
	y = np.ones(10)
	X_ = np.stack([x_] * 10) # 10 * (20*42) * 19
	X_ = X_ + np.random.uniform(0, 20, X.shape)
	y_ = np.zeros(10)
	return X, y, X_, y_

if __name__ == '__main__':
	# gen_sim_data()
	print(max(1, 2))