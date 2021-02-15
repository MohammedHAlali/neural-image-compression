'''
In this code, we'll use the global feature vectors [gfv] that were generated from the autoencoder. Then we'll use thre ML classifiers to do three classification tasks.
'''
import argparse
import os
os.environ['MPLCONFIGDIR'] = '/tmp/'
import glob
import numpy as np
import scipy
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics, utils, svm, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers, Input, callbacks, Sequential, Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
import my_data_utils

print('Numpy version: ', np.__version__)
print('Scipy version: ', scipy.__version__)
print('TF version: ', tf.__version__)
print('sklearn version: ', sklearn.__version__)
parser = argparse.ArgumentParser()
parser.add_argument('model_type', help='svm, rf, or voting')
parser.add_argument('exp_num', help='aeX, huberX, where X > 10')
parser.add_argument('class_type', help='binary, all')
args = parser.parse_args()

print('model type: ', args.model_type)
print('class type: ', args.class_type)
if('ANN' in args.model_type):
	raise ValueError('ERROR: only ML here, you wrote: ', args.model_type)

exp_num = args.exp_num
class_type = args.class_type
model_index = 0
model_type = args.model_type
model_name = model_type+str(model_index)
train_epochs = 20
batch_size = 8
np.set_printoptions(precision=3)

if(class_type == 'all'):
        class_dic = {'bc':0, 'cc':1, 'lc':2, 'pc':3,
                'bn':4, 'cn':5, 'ln':6, 'pn':7}
else:
        class_dic = {'cancer':0, 'normal':1}
class_names = class_dic.keys()

title = '{} classification of {} classes'.format(model_type, len(class_names))
print(title)
#step 1
#my_data_utils.save_data_label('test', class_type, exp_num)
#my_data_utils.save_data_label('train', class_type, exp_num)
#exit()

#step 2
#my_data_utils.convert_to_sparse('train', class_type, exp_num)
#my_data_utils.convert_to_sparse('test', class_type, exp_num)
#my_data_utils.convert_to_sparse('valid', class_type, exp_num)

#step 3
#data, labels = my_data_utils.merge_sparse_data(args.model_type, class_type, exp_num)
#data, labels = my_data_utils.merge_sparse_data('valid', class_type, exp_num)
#data, labels = my_data_utils.merge_sparse_data('test', class_type, exp_num)
#print('loaded sparse data shapes: ', data.shape, labels.shape)
#exit()
upper_out = 'out/{}'.format(exp_num)

out_dir = '{}/{}'.format(upper_out, model_name)
while(os.path.exists(out_dir)):
	model_index += 1
	model_name = args.model_type+str(model_index)
	out_dir = 'out/{}/{}'.format(exp_num, model_name)

if(not os.path.exists(out_dir)):
	os.mkdir(out_dir)
	print('folder created: ', out_dir)

#sparse data
p = 'data/{}_{}_sparse'.format(exp_num, class_type)
train_sparse_x_path = os.path.join(p, 'train_x_sparse.npz')
train_sparse_y_path = os.path.join(p, 'train_y_sparse.npy')
valid_sparse_x_path = os.path.join(p, 'valid_x_sparse.npz')
valid_sparse_y_path = os.path.join(p, 'valid_y_sparse.npy')
test_sparse_x_path = os.path.join(p, 'test_x_sparse.npz')
test_sparse_y_path = os.path.join(p, 'test_y_sparse.npy')
print('train path: ', train_sparse_x_path)

if(args.model_type == 'sgd'):
    #valid_data_list, valid_labels_list = my_data_utils.get_sparse_batch('valid', class_type, exp_num)
    #print('valid data len={}, valid label len={}, valid_data[0].type={} valid_label[0].type={}'.format(len(valid_data_list), len(valid_labels_list), type(valid_data_list[0]), type(valid_labels_list[0])))
    train_data_list, train_labels_list = my_data_utils.get_sparse_batch('train', class_type, exp_num)
    print('train data len={}, train label len={}'.format(len(train_data_list), len(train_labels_list)))
    #DO NOT merge train and valid data

    #for d, l in zip(valid_data_list, valid_labels_list):
    #    train_data_list.append(d)
    #    train_labels_list.append(l)
    #print('train data len={}, train label len={}'.format(len(train_data_list), len(train_labels_list)))
    #test_sparse_x_path = os.path.join('data/{}_{}_sparse'.format(exp_num, class_type), 'test_x_sparse.npz')
    #test_sparse_y_path = os.path.join('data/{}_{}_sparse'.format(exp_num, class_type), 'test_y_sparse.npy')
    valid_sparse_x = scipy.sparse.load_npz(valid_sparse_x_path)
    test_sparse_x = scipy.sparse.load_npz(test_sparse_x_path)
    valid_y = np.load(valid_sparse_y_path).astype('int32')
    test_y = np.load(test_sparse_y_path).astype('int32')
    print('loaded test sparse data of shape x={},y={} type={}'.format(test_sparse_x.shape,test_y.shape, type(test_sparse_x)))
    print('loaded valid sparse data of shape x={},y={} type={}'.format(valid_sparse_x.shape,valid_y.shape, type(valid_sparse_x)))
#check if npy data file exists, load directly, else load from the above method.
elif(os.path.exists(train_sparse_x_path)):
	print('sparse npz data exists')
	print('loading valid path: ', valid_sparse_x_path)
	valid_sparse_x = scipy.sparse.load_npz(valid_sparse_x_path)
	valid_y = np.load(valid_sparse_y_path).astype('int32')
	print('loaded valid sparse data of shape x={},y={} type={}, dtype={}, indices dtype={}, indices shape={}'.format(valid_sparse_x.shape,valid_y.shape, type(valid_sparse_x), valid_sparse_x.dtype, valid_sparse_x.indices.dtype, valid_sparse_x.indices.shape))
	if(valid_sparse_x.indices.dtype == 'int64'):
		#convert dtype of indices from int64 to int32
		v_indices = valid_sparse_x.indices.astype('int32', casting='safe')
		if(np.array_equal(v_indices, valid_sparse_x.indices)):
			print('both indices are equal')
		else:
			raise Exception('ERROR: indices are not equal')
		v_data = valid_sparse_x.data
		v_indptr = valid_sparse_x.indptr
		new_valid_sparse_x = scipy.sparse.csr_matrix((v_data, v_indices, v_indptr), shape=valid_sparse_x.shape)
		print('loaded new valid sparse data of shape x={},y={} type={}, dtype={}, indices dtype={}, indices shape={}'.format(new_valid_sparse_x.shape,valid_y.shape, type(new_valid_sparse_x), new_valid_sparse_x.dtype, new_valid_sparse_x.indices.dtype, new_valid_sparse_x.indices.shape))
	#print('v indices: ', v_indices)
	print('loading train path: ', train_sparse_x_path)
	train_sparse_x = scipy.sparse.load_npz(train_sparse_x_path)
	print('loaded train sparse of shape: ', train_sparse_x.shape)
	train_y = np.load(train_sparse_y_path).astype('int32')
	print('loaded train sparse data of shape x={},y={} type={}, dtype={}, indices dtype={}, indices shape={}'.format(train_sparse_x.shape,train_y.shape, type(train_sparse_x), train_sparse_x.dtype, train_sparse_x.indices.dtype, train_sparse_x.indices.shape))
	#if(train_sparse_x.indices.dtype == 'int64'):
	#	t_indices = train_sparse_x.indices.astype('int32', casting='safe')
	#	if(not np.array_equal(t_indices, train_sparse_x.indices)):
	#		raise Exception('ERROR: new train indices not equal to old')
	#	t_data = train_sparse_x.data
	#	t_indptr = train_sparse_x.indptr
	#	new_train_sparse_x = scipy.sparse.csr_matrix((t_data, t_indices, t_indptr), shape=train_sparse_x.shape)
	#	print('loaded new train sparse data of shape x={},y={} type={}, dtype={}, indices dtype={}, indices shape={}'.format(new_train_sparse_x.shape,train_y.shape, type(new_train_sparse_x), new_train_sparse_x.dtype, new_train_sparse_x.indices.dtype, new_train_sparse_x.indices.shape))
	#	train_sparse_x = new_train_sparse_x
	#if(os.path.exists(valid_sparse_x_path)):
	#	print('valid sparse npz available')
	#	valid_sparse_x = scipy.sparse.load_npz(valid_sparse_x_path)
	#	valid_y = np.load(valid_sparse_y_path).astype('int32')
	#	print('loaded valid data: ', valid_sparse_x.shape, valid_y.shape)
	#	print('trying to merge valid and train data')
	#	train_sparse_x = scipy.sparse.vstack([train_sparse_x, valid_sparse_x])
	#	train_y = np.concatenate([train_y, valid_y])
	#	print('loaded train sparse data of shape x={},y={} type={}, dtype={}, indices dtype={}'.format(train_sparse_x.shape,train_y.shape, type(train_sparse_x), train_sparse_x.dtype, train_sparse_x.indices.dtype))
	test_sparse_x = scipy.sparse.load_npz(test_sparse_x_path)
	test_y = np.load(test_sparse_y_path).astype('int32')
	print('loaded test sparse data of shape x={},y={} type={}'.format(test_sparse_x.shape,test_y.shape, type(test_sparse_x)))
else:
	print('sparse files NOT available')
	valid_x, valid_y = my_data_utils.load_data('valid', class_type, exp_num)
	print('valid data shapes: ', valid_x.shape, valid_y.shape)
	if(valid_x.ndim > 1):
       		shape = valid_x.shape[1:]
       		batch_size = valid_x.shape[0]
       		valid_x = valid_x.reshape((batch_size, np.prod(shape)))
	print('valid_x[0]: ', valid_x[0], ' shape= ', valid_x[0].shape)
	print('valid data shapes: ', valid_x.shape, valid_y.shape)
	test_x, test_y = my_data_utils.load_data('test', class_type, exp_num)
	if(test_x.ndim > 1):
       		shape = test_x.shape[1:]
       		batch_size = test_x.shape[0]
       		test_x = test_x.reshape((batch_size, np.prod(shape)))
	print('test data shapes: ', test_x.shape, test_y.shape)
	train_x, train_y = my_data_utils.load_data('train', class_type, exp_num)
	if(train_x.ndim > 1):
       		shape = train_x.shape[1:]
       		batch_size = train_x.shape[0]
       		train_x = train_x.reshape((batch_size, np.prod(shape)))
	print('train data shapes: ', train_x.shape, train_y.shape)
	print('train_x[0]: ', train_x[0], ' shape= ', train_x[0].shape)
	train_x = np.concatenate([train_x, valid_x]) #row-wise merge, increase num of instances
	train_y = np.concatenate([train_y, valid_y])
	print('after contcatenation: train_x[0]: ', train_x[0] , ' shape= ', train_x[0].shape)
	print('train shapes: ', train_x.shape, train_y.shape)
	#print('valid shapes: ', valid_x.shape, valid_y.shape)
	print('test shapes: ', test_x.shape, test_y.shape)

	train_x, train_y = utils.shuffle(train_x, train_y)
	#valid_x, valid_y = utils.shuffle(valid_x, valid_y)
	test_x, test_y = utils.shuffle(test_x, test_y)

	#train_y = utils.to_categorical(train_y)
	#valid_y = utils.to_categorical(valid_y)
	#test_y is converted to onehot when needed
	print('stats before normalization')
	print('train max: ', np.amax(train_x))
	print('train min: ', np.amin(train_x))
	print('train mean: ', np.mean(train_x))


	#Normalizing the data
	#Standardize features by removing the mean and scaling to unit variance
	#scaler = StandardScaler()

	#This way of normalization changes the sparsity level of the data
	#before this normalization, the sparsity was 95%
	#after it, sparsity is 31%
	#Actually, the data is already normalized to be between 0 and 1
	#maybe we need another method of normalization, but not to decrease sparsity

	#scaler.fit(train_x)
	#train_x = scaler.transform(train_x)
	#valid_x = scaler.transform(valid_x)
	#test_x = scaler.transform(test_x)

	print('calculate sparsity percentage')
	train_non_zero = np.count_nonzero(train_x)
	test_non_zero = np.count_nonzero(test_x)
	train_total_val = np.product(train_x.shape)
	test_total_val = np.product(test_x.shape)
	train_sparsity = (train_total_val - train_non_zero) / train_total_val
	test_sparsity = (test_total_val - test_non_zero) / test_total_val
	train_density = train_non_zero / train_total_val
	test_density = test_non_zero / test_total_val
	print('train sparsity = {}, train desity = {}'.format(train_sparsity, train_density))
	print('test sparsity = {}, test density = {}'.format(test_sparsity, test_density))

	#print('stats after normalization')
	#print('train max: ', np.amax(train_x))
	#print('train min: ', np.amin(train_x))
	#print('train mean: ', np.mean(train_x))
	#print('train shapes: ', train_x.shape, train_y.shape)
	#print('valid shapes: ', valid_x.shape, valid_y.shape)
	#print('test shapes: ', test_x.shape, test_y.shape)
	print('unique labels in train set: ', np.unique(train_y))
	#print('unique labels in valid set: ', np.unique(valid_y))
	print('unique labels in test set: ', np.unique(test_y))
	print('Number of occurences for classes in train set: ', np.bincount(train_y))
	#print('Number of occurences for classes in valid set: ', np.bincount(valid_y))
	print('Number of occurences for classes in test set: ', np.bincount(test_y))

	train_sparse_x = scipy.sparse.csr_matrix(train_x, dtype='float64')
	test_sparse_x = scipy.sparse.csr_matrix(test_x, dtype='float64')
	scipy.sparse.save_npz(os.path.join(upper_out, 'train_x_sparse_{}'.format(class_type)), train_sparse_x)
	np.save(os.path.join(upper_out, 'train_y_sparse_{}'.format(class_type)), train_y)
	scipy.sparse.save_npz(os.path.join(upper_out, 'test_x_sparse_{}'.format(class_type)), test_sparse_x)
	np.save(os.path.join(upper_out, 'test_y_sparse_{}'.format(class_type)), test_y)

	print('sparse data npy saved')
	print('data converted to sparse')

	print('train shape={}, type={}'.format(train_sparse_x.shape, type(train_sparse_x)))
	print('test shape={}, type={}'.format(test_sparse_x.shape, type(test_sparse_x)))


now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H_%M")
#now = now.split()[0]
print('time now is: ', now)
np.set_printoptions(precision=3)

k_reg = regularizers.l2(0.001)

def sgd():
	clf = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False) #all defaults
	return clf

def myLinearSVM():
	clf = svm.LinearSVC(penalty='l1', dual=False, C=0.1, max_iter=-1)
	return clf

def mySVM():
	clf = svm.SVC(probability=True, max_iter=-1, 
                      class_weight='balanced', #uses the values of y to automatically adjust weights 
                                               #inversely proportional to class frequencies
                      tol=0.0001,
			kernel='rbf',
			verbose=False)
	return clf	

def myRF():
	clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                       criterion='gini', max_depth=None, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
	return clf

def myMLP():
	clf = MLPClassifier(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=True, epsilon=1e-08,
              hidden_layer_sizes=(1000, 500, 100, 50), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
              momentum=0.9, n_iter_no_change=200, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='sgd',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
	return clf

def myVoting():
	clf1 = mySVM()
	clf2 = myRF()
	clf3 = myMLP()
	voting_clf = VotingClassifier(
                estimators=[('svm', clf1), ('rf', clf2), ('mlp', clf3)],
                voting='hard')
	return voting_clf

clf = None
if('sgd' in model_name):
	clf = sgd()

elif('svm' in model_name):
	clf = mySVM()
elif('rf' in model_name):
	clf = myRF()
elif('mlp' in model_name):
	clf = myMLP()
elif('voting' in model_name):
	clf = myVoting()
else: 
	raise Exception('ERROR: unknown model_name=', model_name)

print('classifier: ', clf)
'''
############ check if the model works on simple dataset #####################
from sklearn import datasets
toy_X, toy_y = datasets.load_breast_cancer(return_X_y=True)
toy_X_train, toy_X_test, toy_y_train, toy_y_test = train_test_split(
		toy_X, toy_y, random_state=0)
print('shapes of sklearn breast cancer dataset: ', toy_X.shape, toy_y.shape)
toy_clf = clf
toy_clf.fit(toy_X_train, toy_y_train)
print('test score on sklearn breast cancer dataset: ', toy_clf.score(toy_X_test, toy_y_test))
toy_y_pred = clf.predict(toy_X_test)
cm = metrics.confusion_matrix(toy_y_test, toy_y_pred)
print('confusion_matrix of sklearn breast cancer dataset')
print(cm)
'''

print('training on our dataset')
if('sgd' in model_name):
	for ep in range(5):
		print('epoch #:', ep)
		for i in range(len(train_data_list)-1):
			print('batch: ', i)
			#train_batch_x = np.array([train_data_list[i], train_data_list[i+1]])
			batch_x = [train_data_list[i], train_data_list[i+1]]
			train_batch_x = scipy.sparse.vstack(batch_x)
			train_batch_y = np.array([train_labels_list[i], train_labels_list[i+1]])
			print('batch data types: ', type(train_batch_x), type(train_batch_y))
			print('batch data shapes: ', train_batch_x.shape, train_batch_y.shape)
			if(class_type == 'all'):
				classes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
			else:
				classes = np.array([0, 1])
			print('classes: ', classes)
			clf.partial_fit(X=train_batch_x, y=train_batch_y, classes=classes)
else:
	clf.fit(train_sparse_x, train_y)

print('done training')
y_pred = clf.predict(test_sparse_x)

#saving to files to further analysis
np.save(os.path.join(out_dir, 'y_pred'), y_pred)
np.save(os.path.join(out_dir, 'y_true'), test_y)
print('file {}.npy SAVED'.format(os.path.join(out_dir, 'y_pred')))
test_accuracy = clf.score(test_sparse_x, test_y)
print('Accuracy of {} classifier on test set: {:.2f}'.format(model_name, test_accuracy))

cm = metrics.confusion_matrix(test_y, y_pred)
print('confusion_matrix')
print(cm)

#reference:
#https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
tp = np.diag(cm)
fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tn = cm.sum() - (fp + fn + tp)
print('true positive: ', tp)
print('true negative: ', tn)
print('false positive: ', fp)
print('false negative: ', fn)
sen = tp / (tp + fn)
spec = tn / (tn + fp)
print('sensitivity: ', sen, ' avg= ', np.mean(sen))
print('specificity: ', spec, ' avg= ', np.mean(spec))

import seaborn as sns
sns.set(font_scale=1.0) #label size
ax = sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d",cmap='Greys')
title += ', Accuracy=' + str(np.around(test_accuracy, decimals=2))
plt.title(title)
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.show()
img_name = '{}/exp{}_{}_cm_{}.png'.format(out_dir, model_name, exp_num, now)
plt.savefig(img_name, dpi=600)
print('image saved in ', img_name)

print(metrics.classification_report(test_y, y_pred))


print('done')
