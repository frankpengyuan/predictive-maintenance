import numpy as np
import os
import re

vstr = np.vectorize(str)

convert_dir = ["neg_days", "pos_days"]

for mdir in convert_dir:
	cnt = 0
	for one_dir in os.listdir(mdir):
		if "." in one_dir:
			continue
		for one_file in os.listdir(mdir+'/'+one_dir):
			if one_file.endswith(".npy"):
				full_dir = os.path.join(mdir, one_dir, one_file)
				print "converting" + full_dir
				save_dir = os.path.join(mdir, one_dir, one_dir+"_"+one_file)
				save_dir = re.sub(".npy", ".csv", save_dir)
				cnt += 1
				'''
				data = np.load(full_dir)
				str_data = vstr(data)
				with open(save_dir,'wb') as ofile:
					np.savetxt(ofile, str_data, delimiter=",", fmt='%s')
				'''
	print mdir, cnt
