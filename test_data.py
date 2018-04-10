import pathlib
import numpy as np
import pandas as pd

from cv2 import imread
from skimage.color import rgb2grey
from skimage.filters import threshold_otsu
from scipy import ndimage


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    im_id = im_path.parts[-3]
    im = imread(str(im_path))
    im_gray = rgb2grey(im)
    
    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df




import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


# Set some parameters
IMG_WIDTH = 299
IMG_HEIGHT = 299
IMG_CHANNELS = 3
TRAIN_PATH = './data/stage1_train/'
TEST_PATH = './data/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

def get_train_numpy(numpy_path='./data/numpy/'):
    X_train = np.load(numpy_path+'X_train.npy')
    Y_train = np.load(numpy_path+'Y_train.npy')
    X_val = np.load(numpy_path+'X_val.npy')
    Y_val = np.load(numpy_path+'Y_val.npy')
    return (X_train, Y_train), (X_val, Y_val)

def get_test_numpy(numpy_path='./data/numpy/'):
    return np.load(numpy_path+'x_test.npy')


def get_train_data(train_path=TRAIN_PATH, greyscale=True):
        
    train_ids = next(os.walk(TRAIN_PATH))[1]
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        if greyscale:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            img = rgb2grey(img)
            img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 1))
        else:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img / 255.0
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask / 255.0
    return X_train, Y_train


def get_test_data(test_path=TEST_PATH, greyscale=True):
    test_ids = next(os.walk(test_path))[1]
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        if greyscale:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            img = rgb2grey(img)
            img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 1))
        else:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img / 255.0
    return X_test



if __name__ == '__main__':
    #testing = pathlib.Path('data/stage1_test/').glob('*/images/*.png')
    #df = analyze_list_of_images(list(testing))
    #df.to_csv('submission.csv', index=None)
    
    X_train, Y_train = get_train_data()
    
    # Check if training data looks all right
    ix = random.randint(0, len(X_train))
    img = X_train[ix].reshape((IMG_HEIGHT, IMG_WIDTH))
    imshow(img, cmap='Greys')
    plt.show()
    img = Y_train[ix].reshape((IMG_HEIGHT, IMG_WIDTH))
    imshow(np.squeeze(img), cmap='Greys')
    plt.show()
    
    