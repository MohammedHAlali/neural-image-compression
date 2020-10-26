'''
This code reads patches of a single WSI and outputs two (similar but not idential) global feature vectors for the whole WSI. The idea is to convert the whole WSI into a global feature vector of size 150x150x128. Then use the global feature vector data to train classifiers.
'''
import shutil
import argparse
import glob
import os
import multiprocessing as mlt
from sklearn.manifold import TSNE
from image_slicer import slice
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing import image
print('tf version: ', tf.__version__)

def get_pairs(case_path):
	#print('folder_path: ', case_path)
	filenames = os.listdir(case_path)
	#print('number of images to plot = ', len(filenames))
	x_indices = []
	y_indices = []
	x_y_pairs = []
	for i, f in enumerate(filenames):
		#example filename = TCGA-2L-AAQA-11A-01-TSA_region05-44_avg151_r158_g132_b161.png
		#print('[{}/{}]: {}'.format(i, len(filenames), f))
		if('region' not in f):
			continue
		r_index = f.index('region') #get x and y of all images, get maximum dimension
		s_index = f.index('-', r_index)
		u_index = f.index('_', s_index)
		#print('r={}, s={}, u={}'.format(r_index, s_index, u_index))
		x = f[r_index+6:s_index]
		y = f[s_index+1:u_index]
		x_y_pairs.append((x, y))
		#print('x={}, y={}'.format(x, y))
		if(not x in x_indices):
			x_indices.append(x)
		if(not y in y_indices):
			y_indices.append(y)
	print('x_indices=', x_indices)
	x_dim = np.max(np.array(x_indices).astype('int'))
	print('y_indices=', y_indices)
	y_dim = np.max(np.array(y_indices).astype('int'))
	if(x_dim > 200 or y_dim > 200):
		print('ERROR: found big max, x_dim={}, y_dim={}'.format(x_dim, y_dim))
		exit()
	return x_y_pairs

def slicer(image_name, slice_size, out_dir):
	'''
	Take a big image, slice it into multiple mini images of size 128x128, saves them in a subfolder
	modified from: 
	https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
	'''
	#check if slices have been created previously
	#convert out_dir to new location for 128-sized images
	src = out_dir
	splitted = os.path.normpath(out_dir).split(os.path.sep)
	temp = splitted[5]
	splitted[5] = temp+'128'
	#print('\t splitted: ', splitted)
	dest = '/'
	for d in splitted[1:]:
		dest = os.path.join(dest, d)
		#print('dest: ', dest)
		if(not os.path.exists(dest)):
			os.mkdir(dest)
			#print('directory created: ', dest)
	#print('\tnew dest: ', dest)
	out_dir = dest
	slices_names = glob.glob(os.path.join(out_dir, '*.png'))
	if(len(slices_names) == 64):
		return slices_names
	elif(len(slices_names) > 0):
		raise Exception('ERROR: not complete slices available, len=', len(slices_names))
	elif(len(slices_names) == 0): #no slices available, create them now
		filename, file_ext = os.path.splitext(image_name)
		try:
			big_img = Image.open(image_name)
		except Exception as e:
			print('ERROR: could not open image name: ', image_name)
		#print('opened image {} of size {}'.format(image_name, big_img.size))
		width, height = big_img.size
		slices_names = []
		for i in range(0, height//slice_size):
			for j in range(0, width//slice_size):
				box = (j*slice_size, i*slice_size, (j+1)*slice_size, (i+1)*slice_size)
				#print('\t[{},{}]'.format(i, j))
				s = big_img.crop(box)
				if(s.size != (slice_size, slice_size)):
					raise Exception('ERROR: slice size is not correct, size=', s.size)
				out_name = '{}{}_i{}_j{}{}'.format(out_dir,
					filename[filename.rindex('/'):-10],i,j,file_ext)
				#print('\tout_name: ', out_name)
				if('/color_normalized/' in out_name):
					raise Exception('Error: trying to write image in wrong path=', out_name)
					exit()
				s.save(out_name)
				slices_names.append(out_name)
				#print('\tsaved slice in ', out_name)
	return slices_names

def combine_features(class_name, case_id_paths, model_path, out_dir, encoding_size=128):
	pid = os.getpid()
	print('process id: ', pid)
	print('loading model from: ', model_path)
	encoder = load_model(model_path, compile=False)
	encoder.summary()
	#for l in encoder.layers:
	#	print('layer: ', l)
	#	print('type: ', type(l))
	#	print('input shape: ', l.input_shape)
	#	for i, sh in enumerate(l.input_shape):
	#		print(i, '- input shape: ', sh)
	# Check if encoder accepts 128x128 patches
	print('encoder.layer[0]: ', encoder.layers[0])
	print('input shape: ', encoder.layers[0].input_shape[0][1])
	if encoder.layers[0].input_shape[0][1] == 64:
		encoder = add_downsample_to_encoder(encoder)
		print('add 64 sized layer')
		encoder.summary()
	elif encoder.layers[0].input_shape[0][1] == 128:
		pass
	else:
		raise Exception('Model input size not supported.')
	encoder.summary()
	for i,case_path in enumerate(case_id_paths):
		print('[{}/{}]: {}'.format(i, len(case_id_paths), case_path))
		x_y_pairs = get_pairs(case_path)
		files_list = os.listdir(case_path)
		print('Number of images in this case: ', len(files_list))
		first_image = files_list[0] #get all unique patient ids
		#print('first image: ', files_list[0])
		img_id = first_image[:30]
		#print('image_id: ', img_id)
		for k in range(3): #generate three times as many data
			print('-------------Generate iteration#', k, ' ------pid=',pid,'-------------')
			#check if npy file already exists
			file_out_name = 'gfv_{}_{}{}'.format(class_name, img_id[:-6], k)
			file_out_path = os.path.join(out_dir, file_out_name)
			if(os.path.exists(file_out_path+'.npy')):
				print('file exists: ', file_out_path)
				continue
			else:
				print('file does NOT exists')
			global_feature_vector = np.ones((200*8, 200*8, encoding_size))*np.nan
			print('global feature vector shape: ', global_feature_vector.shape)
			img_number_of_features = 0
			for i, (x,y) in enumerate(x_y_pairs):
				image_name = '{}/*region{}-{}_*.png'.format(case_path, x, y)
				#print('[{}/{}] image name: {}'.format(i, len(x_y_pairs), image_name))
				patches = glob.glob(image_name)
				#print('patches: ', patches)
				#if this patch of location [i,j] is actually available
				if(len(patches) == 0):
					# no patch/tile found in this location
					# this means no tissue in this location found
					continue
				name = patches[0]
				print('\t[{}/{}] : {}'.format(i, len(x_y_pairs), name))
				name_only, ext = os.path.splitext(name)
				slices_names = slicer(name, 128, os.path.join(case_path, name_only))
				if(len(slices_names) != 64):
					raise Exception('ERROR: irregular number of slices = ', len(slices_names))
				
				mini_feature_vector = np.ones((8, 8, encoding_size)) * np.nan
				for ii, s in enumerate(slices_names):
					#print('\t\t[{}/{}] slice: {}'.format(ii, len(slices_names), s[41:]))
					x_index = int(s[-8:-7])
					y_index = int(s[-5:-4])
					#print('\t\tx={}, y={}'.format(x_index, y_index))
					index = s.index('TCGA')
					img = image.load_img(s)
					img = image.img_to_array(img)
					img = img / 255.
					img = np.expand_dims(img, axis=0)
					slice_name = s[index:-4]
					an_encoding = encoder.predict(img).squeeze()
					amax = np.amax(an_encoding)
					amin = np.amin(an_encoding)
					amean = np.mean(an_encoding)
					#print('\t\tencoding shape: ', an_encoding.shape)
					#print('\t\tmin={:.4f}, mean={:.4f}, max={:.4f}'.format(amin, amean, amax))
					if(an_encoding.shape[0] != (encoding_size)):
						raise Exception('\t\tERROR: encoding shape not 64x64, but shape=', an_encoding.shape)
					target = mini_feature_vector[x_index,y_index]
					if(not np.isnan(np.sum(target))): 
						#make sure that the position is empty before inserting
						#otherwise there is an error
						raise Exception('\t\tERROR: not empty element: [x={},y={}]={}:'.format(x_index,y_index, target))
					#fill the empty mini feature vector with x, y indices
					#put the encoding back to its original position
					mini_feature_vector[x_index, y_index] = an_encoding
					
					'''
					This baseline method transfers 128x128 tiles to 128 vector
					We transfer 1024x1024 to 128 vector
					To use this baseline method, we need to transfer our 1024x1024 tile to
					64 tiles of size 128x128, then we'll have 64 vectors of size 128 instead of 1 vector of size 128 in our method. So the global feature vector here will be of size [200*64=12800, 12900, 64]
					'''
				
				out_x_index = int(x)*8
				out_y_index = int(y)*8
				mini_feature_vector[np.isnan(mini_feature_vector)] = 0
				print('done, mini feature vector shape: ', mini_feature_vector.shape)
				for i_mini in range(8):
					for j_mini in range(8):
							#print('\tassigning gfv[{},{}] = mini[{},{}]'.format(out_x_index+i_mini, out_y_index+j_mini, i_mini, j_mini))
							global_feature_vector[
							out_x_index+i_mini,
							out_y_index+j_mini] = mini_feature_vector[i_mini, j_mini]
				print('done inserting mini feature in gfv')			
				#shape1 = global_feature_vector[out_x_index,out_y_index].shape
				#shape2 = mini_feature_vector.shape
				#if(shape1 != shape2):
				#	raise Exception('\tERROR: not equal shapes: {}!={}'.format(shape1, shape2))
				#insert the produced feature vector 
				#to its location within the global feature vector
				#global_feature_vector[out_x_index,out_y_index] = mini_feature_vector
				img_number_of_features += 1
				#if(np.amax(feature_vector) > 1):
				#	raise ValueError('ERROR: feature vector with max={} needs normalization'.format(amax))
			print('finished this gfv')				
			image_id = img_id[:-6]
			print('total number of features in {} is {}'.format(image_id, img_number_of_features))
			file_out_name = 'gfv_{}_{}{}'.format(class_name, image_id, k) #gfv = global_feature_vector
			file_out_path = os.path.join(out_dir, file_out_name)
			# Populate NaNs
			global_feature_vector[np.isnan(global_feature_vector)] = 0
			print('gfv shape: ', global_feature_vector.shape)
			np.save(file_out_path, global_feature_vector) #add class_name
			print(file_out_path, ', saved')
			#plot pairplot of values that are not zero
			z, x, y = global_feature_vector.nonzero()
			df_3d = pd.DataFrame()
			df_3d['x'] = x
			df_3d['y'] = y
			df_3d['z'] = z
			sns.pairplot(df_3d)
			plt.show()
			filename = '{}_pairplot.png'.format(file_out_path)
			plt.savefig(filename, dpi=300)
			plt.clf()
			#print('plotting heatmap')
			gfv_mean = np.mean(global_feature_vector, axis=2)
			#print('gfv 2d shape: ', gfv_2d.shape)
			ax = sns.heatmap(gfv_mean, vmin=0, vmax=1)
			plt.savefig(file_out_path+'_mean.png', dpi=300)
			plt.clf()
			tsne = TSNE()
			x_2d = tsne.fit_transform(gfv_mean)
			plt.scatter(x_2d[:, 0], x_2d[:, 1])
			plt.show()
			figname = '{}_tsne.png'.format(file_out_path)
			plt.savefig(figname)
			plt.clf()
			print(figname , ' was saved')
			#print('heatmap saved')
	print('-----done process:', os.getpid())

#source:
#https://github.com/davidtellez/neural-image-compression/blob/master/featurize_wsi.py
def add_downsample_to_encoder(model):

    """
    Adds downsampling layer to input (useful for BiGAN encoder trained with 64x64 patches).
    """

    input_layer = tf.keras.layers.Input((128, 128, 3))
    x = tf.keras.layers.AveragePooling2D()(input_layer)
    x = model(x)

    encoder = tf.keras.models.Model(inputs=input_layer, outputs=x)

    return encoder

if(__name__ == "__main__"):
	print('main process id: ', os.getpid())
	n_cores = mlt.cpu_count()
	print('number of available cpus: ', n_cores)
	parser = argparse.ArgumentParser()
	parser.add_argument('model_name', help='bigan, constrastive, ..etc')
	parser.add_argument('phase', help='train, valid, or test')
	parser.add_argument('class_index', type=int, help='from 0 to 7')
	args = parser.parse_args()
	print('phase = ', args.phase)
	print('class_index = ', args.class_index)
	print('model_name = ', args.model_name)
	phase = args.phase
	model_name = args.model_name
	out_dir = os.path.join('data', model_name)
	if(not os.path.exists(out_dir)):
		os.mkdir(out_dir)
	print('folder created: ', out_dir)
	out_dir = os.path.join(out_dir, phase)
	if(not os.path.exists(out_dir)):
		os.mkdir(out_dir)
	print('folder created: ', out_dir)
	model_path = 'models/encoders_patches_pathology/encoder_{}.h5'.format(model_name)
	print('loading model from: ', model_path)
	#autoencoder = load_model(model_path)
	#autoencoder.summary()
	#print('Getting encoder model')
	#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoding').output)
	#encoder.summary()
	class_names = ['breast', 'colon', 'lung', 'panc', 
			'normal_breast', 'normal_colon', 'normal_lung', 'normal_panc']
	c = class_names[args.class_index]
	input_path = '/common/deogun/alali/data/color_normalized/{}/{}_*'.format(args.phase, c)
	out_dir = os.path.join(out_dir, c)
	if(not os.path.exists(out_dir)):
		os.mkdir(out_dir)
	print('output folder:', out_dir)
	print('input path: ', input_path)
	case_id_paths = glob.glob(input_path)
	case_id_length = len(case_id_paths)
	print('number of case ids: ', case_id_length)
	if(len(case_id_paths) == 0):
		raise ValueError('ERROR: no images found')
	num_processes = 0
	#distributing work to 10 or less processes
	if(case_id_length == 0):
		raise ValueError('ERROR: no images found')
	elif(case_id_length < 10 or case_id_length == 19):
		num_processes = case_id_length
	elif(case_id_length >= 10):
		num_processes = 10
	else:
		raise Exception('ERROR: unknown case id length=', case_id_length)
	list_groups = []
	part_length = case_id_length//num_processes
	print('each of the {} processes will get {} case_ids'.format(num_processes, part_length))
	total_length = 0
	for a in range(num_processes):
		#print('part num: ', a)
		list_groups.append(case_id_paths[part_length*a:part_length*(a+1)])
		a_length = len(list_groups[a])
		print('\tlength of group#{} = {}'.format(a, a_length))
		print('\tlist_group[{}] = {}'.format(a, list_groups[a]))
		total_length += a_length
		print('\ttotal length: ', total_length)

	if(total_length != case_id_length):
		raise Exception('ERROR: could not divide cases evently')
	processes_list = []
	for a in range(num_processes):
		p = mlt.Process(target=combine_features, args=(c, list_groups[a],model_path, out_dir, 128))
		processes_list.append(a)
		p.start()
		p.join()
	print('processes: ', processes_list)
	print('----- done main:', os.getpid())
