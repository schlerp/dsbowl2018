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
IMG_WIDTH = 160
IMG_HEIGHT = 160
IMG_CHANNELS = 3
TRAIN_PATH = './data/stage1_train/'
TEST_PATH = './data/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

def get_train_numpy(numpy_path='./data/numpy/'):
    Xt_out = './data/numpy/X_train_grey.npy'
    Yt_out = './data/numpy/Y_train_grey.npy'    
    X_train = np.load(Xt_out)
    Y_train = np.load(Yt_out)
    return X_train, Y_train

def get_test_numpy(numpy_path='./data/numpy/'):
    X_out = './data/numpy/X_test_grey.npy'
    return np.load(X_out)


def get_train_data(train_path=TRAIN_PATH, greyscale=True):
        
    train_ids = next(os.walk(TRAIN_PATH))[1]
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        if greyscale:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant')
            img = rgb2grey(img)
            img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 1))
        else:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant')
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant'), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    return X_train, Y_train


def get_test_data(test_path=TEST_PATH, greyscale=True):
    test_ids = next(os.walk(test_path))[1]
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        if greyscale:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant')
            img = rgb2grey(img)
            img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 1))
        else:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant')
        X_test[n] = img
    return X_test



if __name__ == '__main__':
    #testing = pathlib.Path('data/stage1_test/').glob('*/images/*.png')
    #df = analyze_list_of_images(list(testing))
    #df.to_csv('submission.csv', index=None)
    
    Xt, Yt = get_train_data()

    from skimage.io import imshow
    from matplotlib import pyplot as plt    
    for img, mask in zip(Xt, Yt):
        plt.subplot(1,2,1)
        img = img.reshape(IMG_HEIGHT, IMG_WIDTH)
        img = img / 255
        imshow(img, cmap='Greys')
        plt.subplot(1,2,2)
        mask = mask.reshape(IMG_HEIGHT, IMG_WIDTH)
        mask = mask / 255
        imshow(mask, cmap='Blues')
        plt.show()
        break 
    
    Xt_out = './data/numpy/X_train_grey.npy'
    Yt_out = './data/numpy/Y_train_grey.npy'
    
    print('saving X training data to {}...'.format(Xt_out))
    np.save(Xt_out, Xt)
    
    print('saving X training data to {}...'.format(Yt_out))
    np.save(Yt_out, Yt)    
    
    X = get_test_data()
    
    X_out = './data/numpy/X_test_grey.npy'
    print('saving X test data to {}...'.format(X_out))
    np.save(X_out, X)       
    
    