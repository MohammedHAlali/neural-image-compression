import os
import glob
import numpy as np
from scipy import sparse

files = glob.glob('data/bigan_all_sparse/train/*x*')
for i, f in enumerate(files):
	print('[{}/{}]: {}'.format(i, len(files), f))
	x = sparse.load_npz(f)
	t = x.indices.dtype
	print('loaded x type: ', type(x), ' indices dtype: ', t)
	if(t != 'int32'):
		print('found different dtype')
		break
	
