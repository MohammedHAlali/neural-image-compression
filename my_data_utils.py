'''
All other processing code files should use these helper functions
'''

import os
import glob
import scipy
from scipy import sparse
import numpy as np
from tensorflow import keras
import tensorflow as tf

#source link:
# https://www.kaggle.com/datapsycho/training-large-scale-data-with-keras-and-tf
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, model_type):
        'Initialization'
        self.list_IDs = list_IDs
        self.on_epoch_end()
        self.model_type = model_type
        #print('list_ids len: ', len(list_IDs))
    	
    def __len__(self):
        # is responsible for agetting the total number of .npy files for each epochs
        # 'Denotes the number of batches per epoch'
        return len(self.list_IDs)
    
    def on_epoch_end(self):
        #  is a confusing method to indicate when epoch will end
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)
        #print('indexes: ', self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        # generate the selected file and associated label file by __getitem__
        'Generates data containing batch_size samples'
        # Generate data
        #print('list_IDs_temp: ', list_IDs_temp)
        filename = list_IDs_temp[0]
        print('file: {}'.format(filename))
        ID = filename[filename.rfind('_')+1:]
        #print('fild ID: ', ID)
        x_file_path = os.path.join(filename[:filename.rfind('/')], 'x_'+ID)
        y_file_path = os.path.join(filename[:filename.rfind('/')], 'y_'+ID)
        print('tying to load data: ', x_file_path)
        print('tying to load label: ', y_file_path)
        
        # Store sample
        X = np.load(x_file_path)
        if(self.model_type == 'ANN'):
            X = X.flatten()
        if(X.ndim < 4):
            X = np.expand_dims(X, axis=0)
        print('loaded x shape: ', X.shape)

        # Store class
        y = np.load(y_file_path).astype('int')
        num_classes = 8
        if('binary' in x_file_path):
            num_classes = 2
        y = keras.utils.to_categorical(y, num_classes=num_classes)
        if(y.ndim < 2):
            y = np.expand_dims(y, axis=0)
        print('loaded y shape: ', y.shape)
        #print('label = ', y.item())
        return X, y
    def __getitem__(self, index):
        # is responsible for select 1 item at a time for training from the given list of file location
        # 'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]
        #print('indexes: ', indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print('list_ids_temp: ', list_IDs_temp)

        # Generate data
        Xa, ya = self.__data_generation(list_IDs_temp)

        return Xa, ya

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

# step 1
def save_data_label(phase, class_type, exp_num):
    class_dic = {'breast':0, 'colon':1, 'lung':2, 'panc':3,
		'normal_breast':4, 'normal_colon':5, 'normal_lung':6, 'normal_panc':7}
    class_names = class_dic.keys()
    data_path = 'data/{}/{}'.format(exp_num, phase)
    print('loading data from path: ', data_path)
    out_path = 'data/{}_{}/'.format(exp_num, class_type)
    if(not os.path.exists(out_path)):
        os.mkdir(out_path)
    out_path = 'data/{}_{}/{}'.format(exp_num, class_type, phase)
    if(not os.path.exists(out_path)):
        os.mkdir(out_path)
    print('saving data to path: ', out_path)
    class_names = os.listdir(data_path)
    print('Number of files: ', len(class_names))
    unique_idx = 0
    labels = []
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
        labels.append(label)
        print('labels: ', labels)
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
            label = np.array(label)
            print('label = ', label.item())
            np.save(label_out_path, label)
            print(label_out_path, ' SAVED')
            unique_idx += 1
        all_labels = np.array(labels)
        print('labels: ', all_labels)
    u = np.unique(all_labels)
    print('unique labels in labels set: ', u)
    if(class_type == 'all' and u.shape[0] < 8):
        raise Exception('ERROR: did not get enough labels')
    print('---- done ----')


# step 2
def convert_to_sparse(phase, class_type, exp_num):
    print('trying to convert dataset to sparse')
    in_path = 'data/{}_{}/{}'.format(exp_num, class_type, phase)
    out_path = 'data/{}_{}_sparse'.format(exp_num, class_type)
    if(not os.path.exists(out_path)):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, phase)
    if(not os.path.exists(out_path)):
        os.mkdir(out_path)
    print('reading data from: ', in_path)
    filenames = glob.glob(os.path.join(in_path, 'x*.npy'))
    data_len = len(filenames)
    print('Number of files read is ', data_len)
    labels = [] #keep track of labels
    for i, filename in enumerate(filenames):
        print('[{}/{}] {}'.format(i, len(filenames), filename))
        _index = filename.rfind('_')
        dot_index = filename.find('.')
        ID = filename[_index+1:dot_index]
        #print('found id: ', ID)
        save_path = os.path.join(out_path, 'sparse_x_{}'.format(ID))
        x_index = filename.find('x')
        y_filename = filename[:x_index]+'y'+filename[x_index+1:]
        print('trying to open y file: ', y_filename)
        y = np.load(y_filename).astype('int')
        labels.append(y.item())
        print('labels = : ', labels)
        if(os.path.exists(save_path+'.npz')):
            print('file EXISTS: ', save_path)
            continue
        ar = np.load(filename)
        print('loaded array shape: ', ar.shape)
        if(sparse.issparse(ar)): #check if sparse
            print('array is sparse')
            continue
        if(ar.ndim > 2):
            ar = ar.flatten()
        print('array flatten shape: ', ar.shape)
        #convert to sparse
        sp = sparse.csr_matrix(ar, dtype='float64')
        #save sparse in out_path 
        #save_path = os.path.join(out_path, 'sparse_x_{}'.format(ID))
        print('trying to save file: ', save_path)
        sparse.save_npz(save_path, sp)
        print('saved')
        #print('label = ', y.item())
        y_save_path = os.path.join(out_path, 'sparse_y_{}'.format(ID))
        print('trying to save y file: ', y_save_path)
        np.save(y_save_path, y)
    all_labels = np.array(labels)
    u = np.unique(all_labels)
    print('unique labels in labels set: ', u)
    if(class_type == 'all' and u.shape[0] < 8):
        raise Exception('ERROR: did not get enough labels')
    print('------ done -----')

# step 3
def merge_sparse_data(phase, class_type, exp_num):
    in_data = 'data/{}_{}_sparse/{}'.format(exp_num, class_type, phase)
    out_path = 'data/{}_{}_sparse/'.format(exp_num, class_type)
    in_path = os.path.join(in_data, '*x*.npz')
    print('trying to find files in path: ', in_path)
    x_filenames = glob.glob(in_path)
    print('Number of data files found: ', len(x_filenames))
    data = []
    labels = []
    for i, f in enumerate(x_filenames):
        print('[{}/{}] {}'.format(i, len(x_filenames), f))
        ar = sparse.load_npz(f)
        print('loaded sparse ar of shape: ', ar.shape)
        index = f.index('x')
        y_filename = f[:index]+'y'+f[index+1:-1]+'y'
        print('trying to load y filename: ', y_filename)
        y_ar = np.load(y_filename).astype('int')
        print('loaded y shape: ', y_ar)
        data.append(ar)
        labels.append(y_ar.item())
        print('labels: ', labels)
    print('data len: ', len(data))
    print('labels len: ', len(labels))
    sparse_data = sparse.vstack(data)
    sparse_labels = np.array(labels)
    print('Number of occurences for classes in label set: ', np.bincount(sparse_labels))
    u = np.unique(sparse_labels)
    print('unique labels in label set: ', u)
    if(class_type == 'all' and u.shape[0] < 8):
       raise Exception('ERROR: did not get enough labels')
    print('sparse data shape: ', sparse_data.shape)
    print('sparse labels shape: ', sparse_labels.shape)
    data_out_path = os.path.join(out_path, '{}_x_sparse'.format(phase))
    labels_out_path = os.path.join(out_path, '{}_y_sparse'.format(phase))
    sparse.save_npz(data_out_path, sparse_data)
    np.save(labels_out_path, sparse_labels)
    print('saved data in : ', data_out_path)
    print('saved labels in : ', labels_out_path)
    return sparse_data, sparse_labels

if(__name__ == "__main__"):
convert_to_sparse( 
