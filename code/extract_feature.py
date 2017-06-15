#! require Python 2.7 environment
import numpy as np
import time
import os
from multiprocessing import Pool, Lock

MAX_THREADS = 6

'''
dtype1=np.dtype([("1", np.int32), ("2", np.float16), ("3", np.uint8), ("5", np.uint8), ("_", np.float16)])
# Or
dtype2=np.dtype('int32,float16,uint8,uint8,float16')
b = np.array([[162, -1.2, 0, 0, 33]] * 3600 * 9 * 700, dtype=np.float32)
print np.mean(b, axis=0)
b = map(tuple, b)
b = np.array(b, dtype=dtype1)
print b.dtype
c = np.array(b.tolist(), dtype=np.float32)
print c.dtype
print c.shape
'''

working_dir = "E:\\cs341-project\\iotdata"
in_root_dir = "E:\\cs341-project\\iotdata\\npz_data_all"
out_root_dir = 'E:\\cs341-project\\iotdata\\npz_feats_all'

def gen_data():
	sim_data1 = np.array([[0, -1.2, 0, 0, 33] * 3] * (3600 // 5) * 24, dtype=np.float32)
	sim_data2 = np.array([[1, 1.0, 0, 0, 0] * 3] * (3600 // 5) * 24, dtype=np.float32)
	sim_heater = [sim_data1] * 400 + [sim_data2] * 300
	np.savez_compressed("b", *sim_heater)

def func_axis0(input):
	f, data = input
	return f(data, axis=0)

def func_axis0_64(input):
	f, data = input
	return f(data, axis=0, dtype=np.float64)

def feature_helper(func, hour_data_list, use64=False):
	if use64:
		return np.concatenate(map(func_axis0_64, zip([func] * len(hour_data_list), hour_data_list)))
	else:
		return np.concatenate(map(func_axis0, zip([func] * len(hour_data_list), hour_data_list)))

def get_change_cnt(ndarry, axis=0):
	a = ndarry[1:,:]
	b = ndarry[:-1,:]
	return np.sum(a != b, axis=axis)

def extract_feats(fname):
	f = np.load(fname)
	keys = sorted([key for key in f], key=lambda x: int(x.split("_")[1]))
	all_data = [f[k] for k in keys]
	all_res = []
	for one_day_raw in all_data:
		#Time at col 0, label at col 1, data start at col 2:
		one_day = one_day_raw[:,2:]
		label = one_day_raw[0, 1]
		features = [np.array([label])]

		hour_data = np.array_split(one_day, 24)
		features.append(feature_helper(np.mean, hour_data, True))
		features.append(feature_helper(np.min, hour_data))
		features.append(feature_helper(np.max, hour_data))
		features.append(feature_helper(np.std, hour_data, True))
		features.append(feature_helper(get_change_cnt, hour_data))
		features.append(np.concatenate([h[0,:] for h in hour_data]))
		features.append(np.concatenate([h[-1,:] for h in hour_data]))

		features.append(np.mean(one_day, axis=0, dtype=np.float64))
		features.append(np.min(one_day, axis=0))
		features.append(np.max(one_day, axis=0))
		features.append(np.std(one_day, axis=0, dtype=np.float64))
		features.append(get_change_cnt(one_day, axis=0))
		features.append(one_day[0, :])
		features.append(one_day[-1, :])

		res = np.reshape(np.concatenate(features), (1, -1))
		all_res.append(res)

	r_table = np.concatenate(all_res, axis=0)
	return r_table

def extract_feats_wrapper(idx, file_name, total_len):
	print ("(%d/%d) start processing %s" % (idx, total_len, file_name))
	tic = time.clock()
	f_feats = extract_feats(file_name)
	pos_days = int(sum(f_feats[:,0]))
	ofname = str(pos_days) + "." + (os.path.basename(file_name)).split('.')[0] + ".feats"
	np.savez_compressed(os.path.join(out_root_dir, ofname), f_feats)
	toc = time.clock()
	print ("(%d/%d)[%ds] finished %s" % (idx, total_len, toc - tic, file_name))
	return 0

def init_child(lock_):
	global lock
	lock = lock_

def extract_multi_files(all_files):
	args = zip(range(len(all_files)), all_files, [len(all_files) - 1] * len(all_files))
	thread_pool = Pool(processes=MAX_THREADS, initializer=init_child, initargs=(lock,))
	results = [thread_pool.apply_async(extract_feats_wrapper, t) for t in args]
	for r in results:
		r.get()

if __name__ == '__main__':
	os.chdir(working_dir)
	all_files = []
	lock = Lock()
	for one_file in os.listdir(in_root_dir):
		if one_file.endswith('.npz'):
			all_files.append(os.path.join(in_root_dir, one_file))
	#gen_data()
	tic_total = time.clock()
	extract_multi_files(all_files)
	toc_total = time.clock()
	print "[%ds] TASK done!" % (toc_total - tic_total)
