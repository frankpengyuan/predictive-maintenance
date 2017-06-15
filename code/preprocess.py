import numpy as np
import os
from collections import deque
import time
from multiprocessing import Pool, Lock
import copy

MAX_THREADS = 6

data_root_dir = "E:\cs341-project\iotdata"
data_dirs = ["BG_Data_Part2", "BG_Data_Part3", "BG_Data_Part5", "BG-Data_Part11", "BG-Data_Part12"]
output_file = "sim_data.csv"
clear_output_file = True
# col start from 0
# Operating_status:_Error_Locking
target_col = 27
neg_sample = "1"

o_fields = '''
file_index
target
Date
Time
Actual_Power
Number_of_burner_starts
Operating_status:_Central_heating_active
Operating_status:_Hot_water_active
Operating_status:_Flame
Relay_status:_Gasvalve
Relay_status:_Fan
Relay_status:_Ignition
Relay_status:_CH_pump
Relay_status:_internal_3-way-valve
Relay_status:_HW_circulation_pump
Supply_temperature_(primary_flow_temperature)_setpoint
Supply_temperature_(primary_flow_temperature)
CH_pump_modulation
Maximum_supply_(primary_flow)_temperature
Hot_water_temperature_setpoint
Hot_water_outlet_temperature
Actual_flow_rate_turbine
Fan_speed
'''
input_cols = [target_col] + [0, 1, 11, 14, 22, 23, 25] + list(range(35, 41)) + [46, 47, 48, 56, 71, 72, 102, 107]

# 15 days buffer
buffer_len = 3600*24*15

def extract_input(fields, input_cols):
	in_fields = []
	for idx in input_cols:
		in_fields.append(fields[idx])
	return in_fields

def store_sample(o_fname, idx, buffers):
	with open(o_fname, "a") as mfile:
		for f_buffer in buffers:
			for line_seg in f_buffer:
				new_line = [str(idx)] + line_seg
				mfile.write(",".join(new_line)+"\n")

def process_file(idx, fname, total_len):
	print ("(%d/%d) start processing %s" % (idx, total_len, fname))
	tic = time.clock()
	f_buffer = deque(iterable=[], maxlen=buffer_len)
	buffers = []
	line_cnt = 0
	with open(fname, "r") as mfile:
		for line in mfile:
			if line_cnt == 0:
				line_cnt += 1
				continue
			fields = line.strip("\n").split("\t")
			input_fields = extract_input(fields, input_cols)
			f_buffer.append(input_fields)
			if fields[target_col] == neg_sample:
				print ("(%d/%d) neg sample at line #%d" % (idx, total_len, line_cnt))
				temp = copy.deepcopy(f_buffer)
				buffers.append(temp)
				f_buffer.clear()
			line_cnt += 1
	# save to file
	if len(buffers) > 0:
		lock.acquire()
		tic2 = time.clock()
		store_sample(o_fname, idx, buffers)
		toc2 = time.clock()
		print ("(%d/%d)[%ds] write to output file done." % (idx, total_len, toc2 - tic2))
		lock.release()
	toc = time.clock()
	print ("(%d/%d)[%ds] finished %s with %d lines." % (idx, total_len, toc - tic, fname, line_cnt))
	return line_cnt

def init_child(lock_, o_fname_):
	global lock
	global o_fname
	lock = lock_
	o_fname = o_fname_

if __name__ == '__main__':

	tic_total = time.clock()

	os.chdir(data_root_dir)
	o_fname = output_file
	if clear_output_file:
		with open(o_fname, 'w') as ofile:
			ofile.write(",".join(o_fields.strip().split("\n")) + "\n")
	all_files = []
	for one_dir in data_dirs:
		for one_file in os.listdir(one_dir):
			if one_file.endswith(".csv"):
				all_files.append(os.path.join(one_dir, one_file))

	lock = Lock()
	args = zip(range(len(all_files)), all_files, [len(all_files) - 1] * len(all_files))
	thread_pool = Pool(processes=MAX_THREADS, initializer=init_child, initargs=(lock, o_fname))
	results = [thread_pool.apply_async(process_file, t) for t in args]
	for r in results:
		r.get()
	toc_total = time.clock()
	print "[%ds] TASK done!" % (toc_total - tic_total)
	#	print '\t', r.get()
	#for idx, fname in enumerate(all_files):
	#	process_file(idx, fname, len(all_files))
		#fname = "BG_Data_Part3\\2029720711597129729.csv"
		#break