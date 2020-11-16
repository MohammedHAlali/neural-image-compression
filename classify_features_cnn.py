'''
In this code, we'll use the global feature vectors [gfv] that were generated from the autoencoder. Then we'll use transfer learing for training
'''
import argparse
import os
os.environ['MPLCONFIGDIR'] = '/tmp/'
import glob
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import utils, regularizers, Input
from tensorflow.keras import callbacks
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, experimental
import my_data_utils

parser = argparse.ArgumentParser()
parser.add_argument('model_type', help='shallow, deep')
parser.add_argument('exp_num', help='aeX, huberX, where X > 10')
parser.add_argument('class_type', help='binary, all')
args = parser.parse_args()

print('model type: ', args.model_type)
print('class type: ', args.class_type)
if('ANN' in args.model_type):
	raise ValueError('ERROR: only CNN here, you wrote: ', args.model_type)

exp_num = args.exp_num
class_type = args.class_type
model_index = 0
model_name = args.model_type+'CNN'+str(model_index)
train_epochs = 100
batch_size = 8

title = '{} classification of all 8 classes'.format(model_name[:-1])
print(title)
upper_out = 'out/{}'.format(exp_num)

out_dir = '{}/{}'.format(upper_out, model_name)

np.set_printoptions(precision=3)
while(os.path.exists(out_dir)):
	model_index += 1
	model_name = args.model_type+'CNN'+str(model_index)
	out_dir = 'out/{}/{}'.format(exp_num, model_name)

if(not os.path.exists(out_dir)):
	os.mkdir(out_dir)
	print('folder created: ', out_dir)

train_path = 'data/{}/train'.format(exp_num)
valid_path = 'out/{}/valid'.format(exp_num)
test_path = 'out/{}/test'.format(exp_num)
if(class_type == 'all'):
	class_dic = {'bc':0, 'cc':1, 'lc':2, 'pc':3,
		'bn':4, 'cn':5, 'ln':6, 'pn':7}
else:
	class_dic = {'cancer':0, 'normal':1}

num_classes = None

if(class_type == 'binary'):
	num_classes = 1
elif(class_type == 'all'):
	num_classes = 8
else:
	raise Exception('ERROR: unknown class type: ', class_type)
with tf.device('/cpu:0'):
	valid_x, valid_y = my_data_utils.load_data_per_file('valid', class_type, exp_num)
	print('valid_x, vaid_y shapes={},{}, type={}'.format(valid_x.shape, valid_y.shape, type(valid_x)))
	test_x, test_y = my_data_utils.load_data('test', class_type, exp_num)
	print('test_x, test_y shapes={},{}, type={}'.format(test_x.shape, test_y.shape, type(test_x)))
	train_x, train_y = my_data_utils.load_data('train', class_type, exp_num)
	print('train_x, train_y shapes={},{}, type={}'.format(train_x.shape, train_y.shape, type(train_x)))

#if(class_type == 'all'):
	#print('convert to categorical')
	#train_y = utils.to_categorical(train_y)
	#valid_y = utils.to_categorical(valid_y)
	#test_y is converted to onehot when needed

print('train shapes: ', train_x.shape, train_y.shape)
print('valid shapes: ', valid_x.shape, valid_y.shape)
print('test shapes: ', test_x.shape, test_y.shape)
#print('Number of occurences for classes in train set: ', np.bincount(train_y))
#print('Number of occurences for classes in valid set: ', np.bincount(valid_y))
#print('Number of occurences for classes in test set: ', np.bincount(test_y))

print('train max: ', np.amax(train_x))
print('train min: ', np.amin(train_x))
print('train mean: ', np.mean(train_x))

#Normalization using Keras
# Create a Normalization layer and set its internal state using the training data
#normalizer = experimental.preprocessing.Normalization()
#normalizer.adapt(train_x)
#normalized_data = normalizer(train_x)
#print('done keras normalizer')
#print('train max: ', np.amax(train_x))
#print('train min: ', np.amin(train_x))
#print('train mean: ', np.mean(train_x))

k_reg = regularizers.l2(0.001)

def shallow0(k_reg=k_reg):
	inputs = Input(shape=(200, 200, 128))
	x = normalizer(inputs)
	x = Conv2D(128, (3, 3), kernel_regularizer=k_reg)(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Flatten()(x)
	x = Dense(128, kernel_regularizer=k_reg)(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	if(class_type == 'all'):
		x = Dense(8)(x)
		output = Activation('softmax')(x)
	elif(class_type == 'binary'):
		x = Dense(1)(x)
		output = Activation('sigmoid')(x)
	model = Model(inputs, output)
	return model

def shallow_cnn(k_reg=k_reg):
        model = Sequential(name='shallow_CNN')
        model.add(Input(shape=(200, 200, 128)))
        #model.add(normalizer())
        model.add(Conv2D(128, (3, 3), kernel_regularizer=k_reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), kernel_regularizer=k_reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(128, kernel_regularizer=k_reg))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        if(class_type == 'all'):
                model.add(Dense(8))
                model.add(Activation('softmax'))
        elif(class_type == 'binary'):
                model.add(Dense(1))
                model.add(Activation('sigmoid'))
        return model
'''
def deep_cnn(k_reg=k_reg):
	model = Sequential(name='deep_model')
	model.add(Input(shape=(200, 200, 128)))
	#model.add(normalizer())
	model.add(Conv2D(128, (3, 3), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(128, (3, 3), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (3, 3), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(64, (1, 1), kernel_regularizer=k_reg))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(32, kernel_regularizer=k_reg))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	if(class_type == 'all'):
		model.add(Dense(8))
		model.add(Activation('softmax'))
	elif(class_type == 'binary'):
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
	return model

if('deep' in model_name):
	model = deep_cnn(k_reg)
el
'''
if('shallow' in model_name):
	model = shallow_cnn(k_reg=k_reg)
else:
	raise Exception('ERROR: unkonwn model name = ', model_name)

model.summary()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
#optimizer = 'adam'
'''
Use sparse categorical crossentropy when your classes are mutually exclusive (e.g. when each sample belongs exactly to one class)
https://datascience.stackexchange.com/questions/41921/sparse-categorical-crossentropy-vs-categorical-crossentropy-keras-accuracy
'''
if(class_type == 'binary'):
	loss = 'binary_crossentropy'
else:
	loss = 'sparse_categorical_crossentropy'
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
print('optimizer: ', model.optimizer.get_config())

log_dir = os.path.join(out_dir, "logs")
patience = int(train_epochs*.1)
print('lr will be reduced when no improvement after {} epochs'.format(patience))
print('EarlyStopping will be when no improvement after {} epochs'.format(patience*2))
callbacks = [callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
		callbacks.ReduceLROnPlateau(verbose=1, factor=0.2, patience=patience),
		callbacks.EarlyStopping(verbose=1, patience=patience*2)
		]
#define class_weight to support under-representative classes
class_weight = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1} #to be updated
print('class weight: ', class_weight)
history = model.fit(train_x, train_y, 
			batch_size=batch_size, 
			epochs=train_epochs, 
			validation_data=(valid_x, valid_y), 
			callbacks=callbacks)

print('train history results: ', history.history)
# Plot training & validation loss values
plt.plot(history.history['loss'], 'g--', label='Training Loss')
plt.plot(history.history['val_loss'], '-', label='Validation Loss')
plt.title('Training and Validation Loss, {}'.format(exp_num))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('{}/exp_{}_train_loss_values.png'.format(out_dir, exp_num), dpi=300)
plt.close()
print('{}/exp_{}_train_loss_values.png SAVED'.format(out_dir, exp_num))

model.save('{}/exp_{}_model'.format(out_dir, exp_num))

#if(class_type == 'all'):
#	results = model.evaluate(test_x, utils.to_categorical(test_y))
#else:
#	results = model.evaluate(test_x, test_y)
#test_accuracy = results[1]
#print('test accuracy: ', test_accuracy)
if(class_type == 'binary'):
	y_pred = model.predict(test_x).squeeze().round().astype('int')
else:
	y_pred = model.predict(test_x)
	y_pred = np.argmax(y_pred, axis=1)

print('y pred shape: ', y_pred.shape)
print('y test shape: ', test_y.shape)
print('test_y[0]: ', test_y[0])
print('y_pred[0]: ', y_pred[0])
if(test_y.ndim > 1 or y_pred.ndim > 1):
	raise Exception('ERROR: shapes hast to be 1-D, but got test_y shape={}, y_pred shape={}'.format(test_y.shape, y_pred.shape))

test_accuracy = metrics.accuracy_score(test_y, y_pred)
print('test accuracy score: ', test_accuracy)
cm = metrics.confusion_matrix(test_y, y_pred)
print('confusion matrix: ')
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

sns.set(font_scale=1.0) #label size
ax = sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_dic.keys(), yticklabels=class_dic.keys(), cmap='Greys')
title = args.model_type + ' Accuracy=' + str(np.around(test_accuracy, decimals=2))
plt.title(title)
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.show()
img_name = '{}/exp{}_cnn_cm.png'.format(out_dir, exp_num)
plt.savefig(img_name, dpi=300)
print('image saved in ', img_name)

print(metrics.classification_report(test_y, y_pred))

print('============================ done ================')
