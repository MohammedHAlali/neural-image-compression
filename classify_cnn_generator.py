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
from tensorflow.keras import Sequential, Model, models
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, experimental
import my_data_utils

parser = argparse.ArgumentParser()
parser.add_argument('model_type', help='cnn or ann')
parser.add_argument('exp_num', help='aeX, huberX, where X > 10')
parser.add_argument('class_type', help='binary, all')
parser.add_argument('existing_model', nargs='?', help='name of existing model to load')
args = parser.parse_args()

print('class type: ', args.class_type)
print('existing model name: ', args.existing_model)

exp_num = args.exp_num
class_type = args.class_type
model_type = args.model_type.upper()
model_index = 0
model_name = model_type+str(model_index)
train_epochs = 1
batch_size = 1

title = '{} classification of all 8 classes'.format(model_type)
print(title)

upper_out = 'out/{}'.format(exp_num)

out_dir = '{}/{}'.format(upper_out, model_name)

np.set_printoptions(precision=3)
while(os.path.exists(out_dir) and (len(os.listdir(out_dir)) > 0)):
	model_index += 1
	model_name = model_type+str(model_index)
	out_dir = 'out/{}/{}'.format(exp_num, model_name)

if(args.existing_model is not None):
	print('getting model from: ', args.existing_model)
	out_dir = 'out/{}/{}'.format(exp_num, args.existing_model)

if(not os.path.exists(out_dir)):
	os.mkdir(out_dir)
	print('folder created: ', out_dir)
print('out_dir: ', out_dir)

train_path = 'data/{}_{}/train'.format(exp_num, class_type)
valid_path = 'data/{}_{}/valid'.format(exp_num, class_type)
test_path = 'data/{}_{}/test'.format(exp_num, class_type)
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

def get_x_ids(path):
    x_filenames = glob.glob(os.path.join(path, 'x*'))
    #filenames = [f[:-4] for f in x_filenames]
    return x_filenames

def get_y_ids(path):
    print('processing labels from :', path)
    y_filenames = glob.glob(os.path.join(path, 'y*'))
    #filenames = [f[:-4] for f in y_filenames]
    labels = []
    for f in y_filenames:
        ar = np.load(f).astype('int')
        labels.append(ar)
    print('number of labels = ', len(labels))
    labels = np.array(labels)
    print('unique labels in set: ', np.unique(labels))
    print('Number of occurences for classes in set: ', np.bincount(labels))
    return y_filenames

with tf.device('/cpu:0'):
    partition = {}
    labels = {}
    partition['valid'] = get_x_ids(valid_path)
    partition['test'] = get_x_ids(test_path)
    partition['train'] = get_x_ids(train_path)

    labels['valid'] = get_y_ids(valid_path)
    labels['test'] = get_y_ids(test_path)
    labels['train'] = get_y_ids(train_path)
    
    train_len = len(partition['train'])
    valid_len = len(partition['valid'])
    test_len = len(partition['test'])
    print('valid x ids len: ', valid_len)
    print('test x ids len: ', test_len)
    print('train x ids len: ', train_len)
    #print('valid y ids len: ', len(labels['valid']))
    #print('test y ids len: ', len(labels['test']))
    #print('train y ids len: ', len(labels['train']))
    if(args.existing_model is not None):
        valid_generator = None
        train_generator = None
        test_generator = my_data_utils.DataGenerator(partition['test'], model_type)
    else:
        valid_generator = my_data_utils.DataGenerator(partition['valid'], model_type)
        train_generator = my_data_utils.DataGenerator(partition['train'], model_type)
        test_generator = my_data_utils.DataGenerator(partition['test'], model_type)


k_reg = regularizers.l2(0.001)

def ann():
	model = Sequential(name='shallow_ANN')
	model.add(Input(shape=(1600*1600*128)))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.25))
	if(num_classes == 8):
		activ = 'softmax'
	else:
		activ = 'sigmoid'
	model.add(Dense(num_classes, activation=activ))
	return model

def cnn():
        model = Sequential(name='shallow_CNN')
        model.add(Input(shape=(1600, 1600, 128)))
        #model.add(normalizer())
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        if(class_type == 'all'):
                model.add(Dense(8))
                model.add(Activation('softmax'))
        elif(class_type == 'binary'):
                model.add(Dense(1))
                model.add(Activation('sigmoid'))
        return model

if(model_type == 'CNN' and args.existing_model is None):
	model = cnn()
elif(model_type == 'CNN' and args.existing_model is not None):
	model_path = 'out/{}/{}/exp_{}_best_model.h5'.format(exp_num, args.existing_model, exp_num)
	print('loading model from: ', model_path)
	model = models.load_model(model_path, compile=True)
	
elif(model_type == 'ANN'):
	model = ann()
else:
	raise Exception('ERROR: unknow model type: ', model_type)

model.summary()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
#optimizer = 'adam'
'''
Use sparse categorical crossentropy when your classes are mutually exclusive (e.g. when each sample belongs exactly to one class)
https://datascience.stackexchange.com/questions/41921/sparse-categorical-crossentropy-vs-categorical-crossentropy-keras-accuracy
'''
if(class_type == 'binary'):
	loss = 'binary_crossentropy'
	metric = 'binary_accuracy'
else:
	loss = 'categorical_crossentropy'
	metric = 'categorical_accuracy'

if(args.existing_model == None):
	#if we are not loading an existing model
	model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metric])
	print('optimizer: ', model.optimizer.get_config())

	log_dir = os.path.join(out_dir, "logs")
	patience = int(train_epochs*.1)
	print('lr will be reduced when no improvement after {} epochs'.format(patience))
	print('EarlyStopping will be when no improvement after {} epochs'.format(patience*2))
	callbacks = [callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
		callbacks.ReduceLROnPlateau(verbose=1, factor=0.2, patience=patience),
		callbacks.EarlyStopping(verbose=1, patience=patience*2),
		callbacks.ModelCheckpoint(filepath='{}/exp_{}_best_model.h5'.format(out_dir, exp_num),
					save_best_only=True,
					verbose=1)
		]
	#define class_weight to support under-representative classes
	#class_weight = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1} #to be updated
	#print('class weight: ', class_weight)
	history = model.fit(train_generator, 
			epochs=train_epochs, 
			steps_per_epoch=train_len,
			validation_data=valid_generator,
			validation_steps=valid_len,
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

results = model.evaluate(test_generator)
print('test results: ', results)


if(class_type == 'binary'):
	y_pred = model.predict(test_generator).squeeze().round().astype('int')
else:
	y_pred = model.predict(test_generator)
	y_pred = np.argmax(y_pred, axis=1)

print('y pred shape: ', y_pred.shape)


'''
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
'''
print('============================ done ================')
