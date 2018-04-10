import keras
from keras import Sequential
from keras.layers import InputLayer, Lambda
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D


from test_data import get_train_numpy, get_test_numpy
(X, Y), (Xval, Yval) = get_train_numpy()

#from test_data import get_train_data
#from sklearn.model_selection import train_test_split
#X, Y = get_train_data()
#X, Xval, Y, Yval = train_test_split(X, Y, test_size=0.20)
IMG_CHANNELS = 1


model_version = 'v2'
batch_size = 32
epochs = 100
early_stop_patience = 5
from test_data import IMG_HEIGHT, IMG_WIDTH


def print_to_file(msg):
    with open('./model/model-schlerp-{}.summary'.format(model_version), 'a+') as f:
        print(msg, file=f)

print_to_file('Model Version: {}'.format(model_version))
print_to_file('Batch Size: {}'.format(batch_size))
print_to_file('Epochs: {}'.format(epochs))
print_to_file('Early Stop Patience: {}'.format(early_stop_patience))


def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    #model.add(Lambda(lambda x: x / 255))
    
    # downscale 1
    model.add(Conv2D(16, (7, 7), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(16, (7, 7), strides=(1,1), padding='same', activation='elu'))
    model.add(MaxPool2D())
    
    # downscale 2
    model.add(Conv2D(12, (5, 5), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(12, (5, 5), strides=(1,1), padding='same', activation='elu'))
    model.add(MaxPool2D())
    
    # downscale 3
    model.add(Conv2D(8, (3, 3), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(8, (3, 3), strides=(1,1), padding='same', activation='elu'))
    model.add(MaxPool2D())
    
    
    # central processing
    model.add(Conv2D(32, (7, 7), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(64, (5, 5), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(64, (5, 5), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2D(32, (7, 7), strides=(1,1), padding='same', activation='elu'))
    
    
    # upscale 1
    model.add(Conv2DTranspose(8, (3, 3), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2DTranspose(8, (3, 3), strides=(1,1), padding='same', activation='elu'))
    model.add(UpSampling2D())
    
    # upscale 2
    model.add(Conv2DTranspose(12, (5, 5), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2DTranspose(12, (5, 5), strides=(1,1), padding='same', activation='elu'))
    model.add(UpSampling2D())
    
    # upscale 3
    model.add(Conv2DTranspose(16, (7, 7), strides=(1,1), padding='same', activation='elu'))
    model.add(Conv2DTranspose(16, (7, 7), strides=(1,1), padding='same', activation='elu'))
    model.add(UpSampling2D())
    
    # output sigmoid
    model.add(Conv2D(1, (1, 1), activation='sigmoid'))

    return model



if __name__ == '__main__':
    #from skimage.io import imshow
    #from matplotlib import pyplot as plt    
    #for img, mask in zip(X, Y):
        #plt.subplot(1,2,1)
        #img = img.reshape(IMG_HEIGHT, IMG_WIDTH)
        #img = img / 255
        #imshow(img, cmap='Greys')
        #plt.subplot(1,2,2)
        #mask = mask.reshape(IMG_HEIGHT, IMG_WIDTH)
        #mask = mask / 255
        #imshow(mask, cmap='Blues')
        #plt.show()
    
    model = build_model()
    
    # add optimiser
    from custom_optimisers import NoisyAdam, NoisySGD
    model.compile(optimizer=NoisySGD(), loss='binary_crossentropy')    
    
    train = True
    
    import os
    if os.path.exists('./model/model-schlerp-{}.h5'.format(model_version)):
        print('loading saved model wights...')
        train = False
        model.load_weights('./model/model-schlerp-{}.h5'.format(model_version))
    
    model.summary()
    
    model.summary(print_fn=print_to_file)
    
    # train model
    if train:
        from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
        tb = TensorBoard(batch_size=batch_size)
        es = EarlyStopping(patience=early_stop_patience, verbose=1)
        cp = ModelCheckpoint('./model/model-schlerp-{}.h5'.format(model_version), 
                             verbose=1, save_best_only=True)
        
        model.fit(X, Y, batch_size=batch_size, epochs=epochs, 
                  validation_data=(Xval, Yval), callbacks=[es, cp, tb])
    
    # test model
    from test_data import get_test_data
    Xtest = get_test_data()
    Xpreds = model.predict(Xtest)
    
    # view tests
    from skimage.io import imshow
    from matplotlib import pyplot as plt
    for img, mask in zip(Xtest, Xpreds):
        plt.subplot(1,2,1)
        img = img.reshape(IMG_HEIGHT, IMG_WIDTH)
        img = img
        imshow(img, cmap='Greys')
        plt.subplot(1,2,2)
        mask = mask.reshape(IMG_HEIGHT, IMG_WIDTH)
        mask = mask
        imshow(mask, cmap='Blues')
        plt.show()
