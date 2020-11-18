'''
All other processing code files should use these helper functions
'''

import os
import scipy
import numpy as np
from tensorflow import keras

#source link:
# https://www.kaggle.com/datapsycho/training-large-scale-data-with-keras-and-tf
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        self.on_epoch_end()
    	
    def __len__(self):
        # is responsible for agetting the total number of .npy files for each epochs
        # 'Denotes the number of batches per epoch'
        return int(len(self.list_IDs))


    def __getitem__(self, index):
        # is responsible for select 1 item at a time for training from the given list of file location
        # 'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y 
    
    def on_epoch_end(self):
        #  is a confusing method to indicate when epoch will end
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def __data_generation(self, list_IDs_temp):
        # generate the selected file and associated label file by __getitem__
        'Generates data containing batch_size samples'
        data_loc = 'data/vae_all/train'
        # Generate data
        for ID in list_IDs_temp:
            x_file_path = os.path.join(data_loc, ID)
            y_file_path = os.path.join(data_loc, image_label_map.get(ID))
        
            # Store sample
            X = np.load(x_file_path)
   
            # Store class
            y = np.load(y_file_path)

        return X, y

def load_data_per_file(phase, class_type, exp_num):
    class_dic = {'breast':0, 'colon':1, 'lung':2, 'panc':3,
                'normal_breast':4, 'normal_colon':5, 'normal_lung':6, 'normal_panc':7}
    class_names = class_dic.keys()
    upper_out = 'out/{}'.format(exp_num)
    data_path = 'data/{}/{}'.format(exp_num, phase)

    data_sparse_x_path = os.path.join(upper_out, '{}_sparse_x_2d_{}.npy'.format(phase, class_type))
    data_x_path = os.path.join(upper_out, '{}_x_2d_{}.npy'.format(phase, class_type))
    data_y_path = os.path.join(upper_out, '{}_y_2d_{}.npy'.format(phase, class_type))
    print('data path: ', data_x_path)

    data_list = []
    label_list = []

    if(not os.path.exists(data_x_path)):
        print('npy data NOT exists')
        #load_data
        print('loading data from path: ', data_path)
        class_names = os.listdir(data_path)
        print('Number of files: ', len(class_names))
        for c in class_names:
            class_path = os.path.join(data_path, c)
            if(not os.path.isdir(class_path)):
                continue
            print('getting from class: ', class_path)
            files = os.listdir(class_path)
            np.random.shuffle(files)
            label = int(class_dic[c])
            if(class_type=='binary' and label <= 3):
                label = 0 #cancer
            elif(class_type=='binary' and label <= 7):
                label = 1 #normal
            print('label = ', label)
            for i, f in enumerate(files):
                if(not '.npy' in f):
                    continue
                f = os.path.join(class_path, f)
                print('[{}/{}] file: {} '.format(i, len(files), f))
                arr = np.load(f)
                print('loaded file shape: ', arr.shape)
                if(np.any(np.isnan(arr))):
                    print('yes, it contains NaNs')
                    arr = np.nan_to_num(arr)
                    print('converted NaN to 0')
                print('data_list len: ', len(data_list))
                #print('label_list shape: ', label_list.shape)
                data_list.append(arr)
                label_list.append(label)
            print('finish loading {} files after class {}'.format(len(data_list), c))
        print('done loading all data files: len=', len(data_list))
        data_list = np.array(data_list)
        label_list = np.array(label_list)
        print('data shapes: ', data_list.shape, label_list.shape)
        np.save(data_x_path, data_list)
        np.save(data_y_path, label_list)
        print('npy files saved: ', data_x_path, data_y_path)
    return data_list, label_list

def load_data(phase='train', class_type='all', exp_num='huber11'):
	class_dic = {'breast':0, 'colon':1, 'lung':2, 'panc':3,
		'normal_breast':4, 'normal_colon':5, 'normal_lung':6, 'normal_panc':7}
	class_names = class_dic.keys()
	upper_out = 'out/{}'.format(exp_num)
	data_path = 'data/{}/{}'.format(exp_num, phase)
	
	data_sparse_x_path = os.path.join(upper_out, '{}_sparse_x_2d_{}.npy'.format(phase, class_type))
	data_x_path = os.path.join(upper_out, '{}_x_2d_{}.npy'.format(phase, class_type))
	data_y_path = os.path.join(upper_out, '{}_y_2d_{}.npy'.format(phase, class_type))
	print('data path: ', data_x_path)
	
	data_list = []
	label_list = []

	if(os.path.exists(data_sparse_x_path)):
		print('sparse data available at: ', data_sparse_x_path)
		data_list = scipy.sparse.load_npz(data_sparse_x_path)
		label_list = np.load(data_y_path)
		print('sparse data loaded, shape: ', data_list.shape, type(data_list))
	elif(os.path.exists(data_x_path)):
		#load data
		print('npy data exists at: ', data_x_path)
		data_list = np.load(data_x_path)
		label_list = np.load(data_y_path)
		
	else:
		print('npy data NOT exists')
		#load_data
		print('loading data from path: ', data_path)
		class_names = os.listdir(data_path)
		print('Number of files: ', len(class_names))
		for c in class_names:
			class_path = os.path.join(data_path, c)
			if(not os.path.isdir(class_path)):
				continue
			print('getting from class: ', class_path)
			class_x_path = os.path.join(upper_out, '{}_{}_x_2d_{}.npy'.format(phase, c, class_type))
			class_y_path = os.path.join(upper_out, '{}_{}_y_2d_{}.npy'.format(phase, c, class_type))
			if(os.path.exists(class_x_path)):
				print('file {} exists'.format(class_x_path))
				class_data_list = np.load(class_x_path)
				print('loaded data shape: ', class_data_list.shape)
				class_label_list = np.load(class_y_path)
				print('loaded labels shape: ', class_label_list.shape)
				if(len(data_list) == 0):
					print('total data shape: ', len(data_list))
					data_list = class_data_list
					label_list = class_label_list
				else:
					data_list = np.array(data_list)
					label_list = np.array(label_list)
					print('total data shape: ', data_list.shape)
					data_list = np.concatenate((data_list, class_data_list))
					label_list = np.concatenate((label_list, class_label_list))
					print('concatenated total data shape: ', data_list.shape)
				continue
			else:
				print('file {} NOT exists'.format(class_x_path))
				class_data_list = []
				class_label_list = []
				
				files = os.listdir(class_path)
				np.random.shuffle(files)
				label = int(class_dic[c])
				if(class_type=='binary' and label <= 3):
					label = 0 #cancer
				elif(class_type=='binary' and label <= 7):
					label = 1 #normal
				print('label = ', label)
				for i, f in enumerate(files):
					if(not '.npy' in f):
						continue
					f = os.path.join(class_path, f)
					print('[{}/{}] file: {} '.format(i, len(files), f))
					arr = np.load(f)
					print('loaded file shape: ', arr.shape)
					if(np.any(np.isnan(arr))):
						print('yes, it contains NaNs')
						arr = np.nan_to_num(arr)
						print('converted NaN to 0')
					#print('data_list shape: ', data_list.shape)
					#print('label_list shape: ', label_list.shape)
					class_data_list.append(arr)
					class_label_list.append(label)
				print('finish loading {} files from class {}'.format(len(class_data_list), c))
				class_data_list = np.array(class_data_list)
				class_label_list = np.array(class_label_list)
				print('class data shapes: ', class_data_list.shape, class_label_list.shape)
			np.save(class_x_path, class_data_list)
			np.save(class_y_path, class_label_list)
			print('file {} SAVED'.format(class_x_path))
			print('file {} SAVED'.format(class_y_path))
		#check this step later
		data_list = np.array(data_list)
		label_list = np.array(label_list)
		print('final data shapes: ', data_list.shape, label_list.shape)
		#shuffle both data and labels
		np.save(data_x_path, data_list)
		np.save(data_y_path, label_list)
		print('npy files saved: ', data_x_path, data_y_path)
	return data_list, label_list

def save_data_label(phase, class_type, exp_num):
    class_dic = {'breast':0, 'colon':1, 'lung':2, 'panc':3,
		'normal_breast':4, 'normal_colon':5, 'normal_lung':6, 'normal_panc':7}
    class_names = class_dic.keys()
    upper_out = 'out/{}'.format(exp_num)
    data_path = 'data/{}/{}'.format(exp_num, phase)
    print('loading data from path: ', data_path)
    out_path = 'data/{}_all/{}'.format(exp_num, phase)
    print('saving data to path: ', out_path)
    class_names = os.listdir(data_path)
    print('Number of files: ', len(class_names))
    unique_idx = 0
    for c in class_names:
        class_path = os.path.join(data_path, c)
        if(not os.path.isdir(class_path)):
            continue
        print('getting from class: ', class_path)
        files = os.listdir(class_path)
        np.random.shuffle(files)
        label = int(class_dic[c])
        if(class_type=='binary' and label <= 3):
            label = 0 #cancer
        elif(class_type=='binary' and label <= 7):
            label = 1 #normal
        print('label = ', label)
        for i, f in enumerate(files):
            if(not '.npy' in f):
                continue
            f = os.path.join(class_path, f)
            print('[{}/{}] file: {} '.format(i, len(files), f))
            arr = np.load(f)
            print('loaded file shape: ', arr.shape)
            if(np.any(np.isnan(arr))):
                print('yes, it contains NaNs')
                arr = np.nan_to_num(arr)
                print('converted NaN to 0')
            'save data and label in out_dir with xi.npy and yi.npy'
            data_out_path = os.path.join(out_path, 'x_{}'.format(unique_idx))
            label_out_path = os.path.join(out_path, 'y_{}'.format(unique_idx))
            print('trying to save : ', data_out_path)
            np.save(data_out_path, np.array(arr))
            print(data_out_path, ' SAVED')
            np.save(label_out_path, np.array(label))
            print(label_out_path, ' SAVED')
            unique_idx += 1

