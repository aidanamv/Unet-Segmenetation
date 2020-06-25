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
from model import *
from data_processing import *

print(tf.__version__)
print(tf.executing_eagerly())

img_dir = './dataset/'

# Required image dimensions
output_height = 256
output_width = 256
frame_tensors, masks_tensors, frames_list, masks_list = read_images(img_dir)
# Make an iterator to extract images from the tensor dataset
frame_batches = tf.compat.v1.data.make_one_shot_iterator(frame_tensors)  # outside of TF Eager, we would use make_one_shot_iterator
mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks_tensors)
n_images_to_show = 5

for i in range(n_images_to_show):
    # Get the next image from iterator
    frame = frame_batches.next().numpy().astype(np.uint8)
    mask = mask_batches.next().numpy().astype(np.uint8)
 # Plot the corresponding frames and masks
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(frame)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

DATA_PATH = './dataset/'

# Create folders to hold images and masks
folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val']

for folder in folders:
    try:
        os.makedirs(DATA_PATH + folder)
    except Exception as e: print(e)


generate_image_folder_structure(DATA_PATH,frame_tensors, masks_tensors, frames_list, masks_list)
label_codes=[(255,0,0), (0,255,0), (255,255,0),(0,0,0)]
label_names=['comp_1','comp2_2', 'both','background']

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}
name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}



# Seed defined for aligning images and their masks
seed = 1
model = get_small_unet(n_filters = 4)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[dice_coef_loss,'accuracy'])
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='my_train_1.h5', monitor='accuracy', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='max', monitor='val_accuracy', patience=10, verbose=1)
callbacks = [tb, mc, es]
batch_size = 5
validation_steps = 5
num_epochs = 100
result = model.fit_generator(TrainAugmentGenerator(), steps_per_epoch=20 ,
                validation_data = ValAugmentGenerator(),
                validation_steps = validation_steps, epochs=num_epochs, callbacks=callbacks)
model.save_weights("my_train_1.h5", overwrite=True)
# Get actual number of epochs model was trained for
N = len(result.history['loss'])
#Plot the model evaluation history
plt.style.use("ggplot")
fig = plt.figure(figsize=(20,8))
fig.add_subplot(1,2,1)
plt.title("Training Loss")
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.ylim(0, 1)
fig.add_subplot(1,2,2)
plt.title("Training Accuracy")
plt.plot(np.arange(0, N), result.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), result.history["val_accuracy"], label="val_accuracy")
plt.ylim(0, 1)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
testing_gen = ValAugmentGenerator()
batch_img,batch_mask = next(testing_gen)
print(tf.shape(batch_img))
pred_all= model.predict(batch_img)
np.shape(pred_all)
for i in range(0, np.shape(pred_all)[0]):
    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(batch_img[i])
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(onehot_to_rgb(batch_mask[i], id2code))
    ax2.grid(b=None)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Predicted labels')
    ax3.imshow(onehot_to_rgb(pred_all[i], id2code))
    ax3.grid(b=None)

    plt.show()


test_dir='./test_images/'
test_tensors, test_list = read_test_images(test_dir)
test_batches = tf.compat.v1.data.make_one_shot_iterator(test_tensors)  # outside of TF Eager, we would use make_one_shot_iterator
n_images_to_show = 5
test=np.zeros([5,256,256,3])


for i in range(n_images_to_show):
    # Get the next image from iterator
    test[i,:,:,:]= test_batches.next().numpy().astype(np.uint8)


pred_all=model.predict(test)
for i in range(0, np.shape(pred_all)[0]):
    frame=test_batches.next().numpy().astype(np.uint8)
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(frame)
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Predicted labels')
    ax2.imshow(onehot_to_rgb(pred_all[i], id2code))
    ax2.grid(b=None)
    plt.show()