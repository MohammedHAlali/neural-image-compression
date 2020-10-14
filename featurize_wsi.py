"""
This module runs an encoder over a vectorized WSI to obtain features from it (compress it).
"""

import matplotlib as mpl
mpl.use('Agg')  # plot figures when no screen available
from matplotlib import pyplot as plt
from os.path import basename, dirname, join, exists, splitext
import os
import numpy as np
import random
from scipy.ndimage.morphology import distance_transform_edt
import tensorflow as tf
from tensorflow import keras
#import keras
#from vectorize_wsi import vectorize_wsi
import sys
import glob

def encode_wsi_npy(encoder, input_path, output_path):
    """
    Featurizes a vectorized WSI taking augmentations into account. Augments indexes and patches properly.

    Args:
        encoder: model transforming a patch to a vector code.
        wsi_pattern (str): path pattern pointing to vectorized WSI.
        batch_size (int): number of patches to encode simultaneously by the GPU.
        output_path (str): path pattern to output files.
            For example: /path/normal_001_features.npy'.
        output_preview_pattern (str or None): optional path pattern to preview files.
            For example: /path/normal_001_{f_min}_{f_max}_features.png'.
        output_distance_map (bool): True to write distance map useful to extract image crops.

    """

    # Check if encoder accepts 128x128 patches
    #if encoder.layers[0].input_shape[1] == 64:
    #    encoder = add_downsample_to_encoder(encoder)
    #elif encoder.layers[0].input_shape[1] == 128:
    #    pass
    #else:
    #    raise Exception('Model input size not supported.')

    print('input path: ', input_path)

    # Read wsi
    #wsi_sequence = WsiNpySequence(wsi_pattern=wsi_pattern, batch_size=batch_size)

    # Config
    #xs = wsi_sequence.xs
    #ys = wsi_sequence.ys
    #image_shape = wsi_sequence.image_shape

    # get npy files
    cases = os.listdir(input_path)
    for j, c in enumerate(cases):
        print('[{}/{}]: {}'.format(j, len(cases), c))
        case_path = os.path.join(input_path, c)
        print('case path: ', case_path)
        files = glob.glob(os.path.join(case_path, '*.npy'))
        print('Number of npy files: ', len(files))
        gfv = np.ones((64, 64, 128)) * np.nan
        print('gfv shape: ', gfv.shape)
        for i, f in enumerate(files):
            print('[{}/{}]: {}'.format(i, len(files), f))
            
            x_index = int(f[-8:-7])
            y_index = int(f[-5:-4])
            print('x={}, y={}'.format(x_index, y_index))
            arr = np.load(f)
            print('loaded shape: ', arr.shape)
            arr = np.expand_dims(arr, 0)
            # predict
            a_feature = encoder.predict(arr)
            print('encoding shape: ', a_feature.shape)
            amin = np.amin(a_feature)
            amax = np.amax(a_feature)
            amean = np.mean(a_feature)
            print('min={}, max={}, mean={}'.format(amin, amax, amean))
            # store the feature in the right spatial position
            gfv[x_index, y_index] = a_feature
        print('gfv shape: ', gfv.shape)
    # Store each patch feature in the right spatial position
    #for patch_feature, x, y in zip(patch_features, xs, ys):
    #    features[:, y, x] = patch_feature

    # Populate NaNs
    gfv[np.isnan(gfv)] = 0

    # Save to disk float16
    np.save(output_path, features.astype('float16'))

    # Plot
    if output_preview_pattern:
        plot_feature_map(np.copy(features), output_preview_pattern)

    # Distance map
    if output_distance_map:
        try:
            filename = splitext(basename(output_path))[0]
            output_dm_path = join(dirname(output_path), filename + '_distance_map.npy')
            distance_map = compute_single_distance_map(features.astype('float32'))
            np.save(output_dm_path, distance_map)
        except Exception as e:
            print('Failed to compute distance map for {f}. Exception: {e}.'.format(f=output_path, e=e), flush=True)


class WsiNpySequence(keras.utils.Sequence):

    """
    This class is a Keras sequence used to make predictions on vectorized WSIs.
    """

    def __init__(self, wsi_pattern, batch_size):
        """
        This class is a Keras sequence used to make predictions on vectorized WSIs.

        Args:
            wsi_pattern (str): path pattern pointing to location of vectorized WSI.
                For example: "/path/normal_060_{item}.npy".
            batch_size (int): batch size to process the patches.
        """

        # Params
        self.batch_size = batch_size
        self.wsi_pattern = wsi_pattern

        # Read data
        self.image_tiles = np.load(wsi_pattern.format(item='patches'))
        self.xs = np.load(wsi_pattern.format(item='x_idx'))
        self.ys = np.load(wsi_pattern.format(item='y_idx'))
        self.image_shape = np.load(wsi_pattern.format(item='im_shape'))
        self.n_samples = self.image_tiles.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def get_batch(self, idx):
        """
        Gets batches based on index. The last batch might have smaller length than batch size.
        Args:
            idx: index in batches..

        Returns: batch of image patches in [-1, +1] [b, x, y, ch] format.

        """

        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Build batch
        image_tiles = self.image_tiles[idxs, ...]

        # Format
        image_tiles = (image_tiles / 255.0 * 2) - 1

        return image_tiles

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        return batch


def plot_feature_map(features, output_pattern):
    """
    Preview of the featurized WSI. Draws a grid where each small image is a feature map. Normalizes the set of feature
    maps using the 3rd and 97th percentiles of the entire feature volume. Includes these values in the filename.

    Args:
        features: numpy array with format [c, x, y].
        output_pattern (str): path pattern of the form '/path/tumor_001_90_none_{f_min:.3f}_{f_max:.3f}_features.png'

    """

    # Get range for normalization
    f_min = np.percentile(features[features != 0], 3)
    f_max = np.percentile(features[features != 0], 97)

    # Detect background (estimate)
    features[features == 0] = np.nan

    # Normalize and clip values
    features = (features - f_min) / (f_max - f_min + 1e-6)
    features = np.clip(features, 0, 1)

    # Add background
    features[features == np.nan] = 0.5

    # Make batch
    data = features[:, np.newaxis, :, :].transpose(0, 2, 3, 1)

    # Make grid
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.0)
    padding = ((0, 0), (5, 5), (5, 5)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.5)

    # Tile the individual thumbnails into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # Map the normalized data to colors RGBA
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1)
    image = cmap(norm(data[:, :, 0]))

    # Save the image
    plt.imsave(output_pattern.format(f_min=f_min, f_max=f_max), image)


def compute_single_distance_map(features):
    """
    Computes distance to tissue map. It is useful to detect where the tissue areas are located and take crops from them.

    :param features: featurized whole-slide image.
    :return: distance map array
    """

    # Binarize
    features = features.std(axis=0)
    features[features != 0] = 1

    # Distance transform
    distance_map = distance_transform_edt(features)
    distance_map = distance_map / np.max(distance_map)
    distance_map = np.square(distance_map)
    distance_map = distance_map / np.sum(distance_map)

    return distance_map

def add_downsample_to_encoder(model):

    """
    Adds downsampling layer to input (useful for BiGAN encoder trained with 64x64 patches).
    """

    input_layer = keras.layers.Input((128, 128, 3))
    x = keras.layers.AveragePooling2D()(input_layer)
    x = model(x)

    encoder = keras.models.Model(inputs=input_layer, outputs=x)

    return encoder

if __name__ == '__main__':
    #run as: 
    # python featurize_wsi.py TCGA-A8-A07S-01Z-00-DX1.svs out/ 1000 1 1 models/encoders_patches_pathology/encoder_contrastive.h5 1
    # Paths
    input_path = '/common/deogun/alali/data/color_normalized_npy/train/'
    output_path = 'data/'
    #patch_size = int(sys.argv[4])
    #stride = int(sys.argv[4])
    #downsample = int(sys.argv[5])
    model_path = sys.argv[1]
    #batch_size = sys.argv[4]
    #filename = splitext(basename(image_path))[0]
    #output_pattern = join(output_dir, filename + '_{item}.npy')
    #output_path = join(output_dir, filename + '_features.npy')
    #output_preview_pattern = join(output_dir, filename + '_{f_min}_{f_max}_features.png')

    # Vectorize slide
    vectorize_wsi(
        image_path=image_path,
        output_pattern=output_pattern,
        patch_size=patch_size,
        stride=stride,
        downsample=downsample
    )
    
    print('model path: ', model_path)
    # Load encoder model
    encoder = keras.models.load_model(
        filepath=model_path
    )
    encoder.summary()
    # Featurize (encode) image
    encode_wsi_npy(
        encoder=encoder,
        input_path=input_path,
        #batch_size=batch_size,
        output_path=output_path,
        #output_preview_pattern=None,
        #output_distance_map=True
    )
