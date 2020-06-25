import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from PIL import Image
from pylab import *
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

label_codes=[(255,0,0), (0,255,0), (255,255,0),(0,0,0)]
label_names=['comp_1','comp2_2', 'both','background']
DATA_PATH='./dataset/'
code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}
name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}

data_gen_args = dict(rescale=1./255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)
def _read_to_tensor(fname, output_height=256, output_width=256, normalize_data=False):
    '''Function to read images from given image file path, and provide resized images as tensors
        Inputs:
            fname - image file path
            output_height - required output image height
            output_width - required output image width
            normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
        Output: Processed image tenso
    '''

    # Read the image as a tensor
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings)

    # Resize the image
    output = tf.image.resize(imgs_decoded, [output_height, output_width])

    # Normalize if required
    if normalize_data:
        output = (output - 128) / 128
    return output

#reading frames and masks
def read_images(img_dir):
    '''Function to get all image directories, read images and masks in separate tensors
        Inputs:
            img_dir - file directory
        Outputs
            frame_tensors, masks_tensors, frame files list, mask files list
    '''

    # Get the file names list from provided directory
    file_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # Separate frame and mask files lists, exclude unnecessary files
    frames_list = [file for file in file_list if ('_labeled' not in file) and ('txt' not in file) and ('m' not in file)]
    frames_list.sort()
    masks_list = [file for file in file_list if ('_labeled' in file) and ('txt' not in file) and  ('m' not in file)]
    masks_list.sort()
    print('{} frame files found in the provided directory.'.format(len(frames_list)))
    print('{} mask files found in the provided directory.'.format(len(masks_list)))

    # Create file paths from file names
    frames_paths = [os.path.join(img_dir, fname) for fname in frames_list]
    masks_paths = [os.path.join(img_dir, fname) for fname in masks_list]

    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)

    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)

    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))

    return frame_tensors, masks_tensors, frames_list, masks_list


def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c

def generate_image_folder_structure(DATA_PATH,frames, masks, frames_list, masks_list):
    '''Function to save images in the appropriate folder directories
        Inputs:
            frames - frame tensor dataset
            masks - mask tensor dataset
            frames_list - frame file paths
            masks_list - mask file paths
    '''
    # Create iterators for frames and masks
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(
        frames)  # outside of TF Eager, we would use make_one_shot_iterator
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)

    # Iterate over the train images while saving the frames and masks in appropriate folders
    dir_name = 'train'
    for file in zip(frames_list[:-round(0.2 * len(frames_list))], masks_list[:-round(0.2 * len(masks_list))]):
        # Convert tensors to numpy arrays
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        # Convert numpy arrays to images
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)

        # Save frames and masks to correct directories
        frame.save(DATA_PATH + '{}_frames/{}'.format(dir_name, dir_name) + '/' + file[0])
        mask.save(DATA_PATH + '{}_masks/{}'.format(dir_name, dir_name) + '/' + file[1])

    # Iterate over the val images while saving the frames and masks in appropriate folders
    dir_name = 'val'
    for file in zip(frames_list[-round(0.2 * len(frames_list)):], masks_list[-round(0.2 * len(masks_list)):]):
        # Convert tensors to numpy arrays
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        # Convert numpy arrays to images
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)

        # Save frames and masks to correct directories
        frame.save(DATA_PATH + '{}_frames/{}'.format(dir_name, dir_name) + '/' + file[0])
        mask.save(DATA_PATH + '{}_masks/{}'.format(dir_name, dir_name) + '/' + file[1])

    print("Saved {} frames to directory {}".format(len(frames_list), DATA_PATH))
    print("Saved {} masks to directory {}".format(len(masks_list), DATA_PATH))

def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs:
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs:
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3)
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def TrainAugmentGenerator(seed=1, batch_size=5 ):
    '''Train Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    train_image_generator = train_frames_datagen.flow_from_directory(
        DATA_PATH + 'train_frames/',
        batch_size=batch_size, seed=seed)

    train_mask_generator = train_masks_datagen.flow_from_directory(
        DATA_PATH + 'train_masks/',
        batch_size=batch_size, seed=seed)

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def ValAugmentGenerator(seed=1, batch_size=5):
    '''Validation Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    val_image_generator = val_frames_datagen.flow_from_directory(
        DATA_PATH + 'val_frames/',
        batch_size=batch_size, seed=seed)

    val_mask_generator = val_masks_datagen.flow_from_directory(
        DATA_PATH + 'val_masks/',
        batch_size=batch_size, seed=seed)

    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]
        yield X1i[0], np.asarray(mask_encoded)


def read_test_images(img_dir):
    '''Function to get all image directories, read images and masks in separate tensors
        Inputs:
            img_dir - file directory
        Outputs
            frame_tensors, masks_tensors, frame files list, mask files list
    '''

    # Get the file names list from provided directory
    file_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # Separate frame and mask files lists, exclude unnecessary files
    frames_list = [file for file in file_list if ('_labeled' not in file) and ('txt' not in file) and ('m' not in file)]
    frames_list.sort()
    print('{} frame files found in the provided directory.'.format(len(frames_list)))

    # Create file paths from file names
    frames_paths = [os.path.join(img_dir, fname) for fname in frames_list]

    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)

    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)

    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))

    return frame_tensors, frames_list

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)