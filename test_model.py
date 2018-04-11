import keras
from keras import Model
from keras.layers import Input, Lambda, concatenate
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D


from data_utils import get_train_numpy, get_test_numpy

X, Y = get_train_numpy()
from sklearn.model_selection import train_test_split

X, Xval, Y, Yval = train_test_split(X, Y, test_size=0.20)

#from test_data import get_train_data
#from sklearn.model_selection import train_test_split
#X, Y = get_train_data()
#X, Xval, Y, Yval = train_test_split(X, Y, test_size=0.20)
IMG_CHANNELS = 1


model_version = 'v3'
batch_size = 32
epochs = 1000
early_stop_patience = 10
from data_utils import IMG_HEIGHT, IMG_WIDTH



def print_to_file(msg):
    with open('./model/model-schlerp-{}.summary'.format(model_version), 'a+') as f:
        print(msg, file=f)

print_to_file('Model Version: {}'.format(model_version))
print_to_file('Batch Size: {}'.format(batch_size))
print_to_file('Epochs: {}'.format(epochs))
print_to_file('Early Stop Patience: {}'.format(early_stop_patience))
print_to_file('Image Dimensions: {}h, {}w, {}c'.format(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))


def build_model():

    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    c1 = Conv2D(32, (5, 5), padding='same', activation='elu')(input_layer)
    c1 = Conv2D(32, (5, 5), padding='same', activation='elu')(c1)
    
    mp1 = MaxPool2D()(c1)
    
    c2 = Conv2D(48, (3, 3), padding='same', activation='elu')(mp1)
    c2 = Conv2D(48, (3, 3), padding='same', activation='elu')(c2)
    
    mp2 = MaxPool2D()(c2)   
    
    c3 = Conv2D(64, (3, 3), padding='same', activation='elu')(mp2)
    c3 = Conv2D(64, (3, 3), padding='same', activation='elu')(c3)
    
    mp3 = MaxPool2D()(c3)
    
    
    cp = Conv2D(96, (3, 3), padding='same', activation='elu')(mp3)
    cp = Conv2D(128, (3, 3), padding='same', activation='elu')(cp)
    cp = Conv2D(96, (3, 3), padding='same', activation='elu')(cp)
    
    
    uc1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(cp)
    uc1 = concatenate([uc1, c3])
    uc1 = Conv2D(64, (3, 3), padding='same', activation='elu')(uc1)
    uc1 = Conv2D(64, (3, 3), padding='same', activation='elu')(uc1)
    
    uc2 = Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same')(uc1)
    uc2 = concatenate([uc2, c2])
    uc2 = Conv2D(48, (3, 3), padding='same', activation='elu')(uc2)
    uc2 = Conv2D(48, (3, 3), padding='same', activation='elu')(uc2)
    
    uc3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(uc2)
    uc3 = concatenate([uc3, c1])
    uc3 = Conv2D(32, (3, 3), padding='same', activation='elu')(uc3)
    uc3 = Conv2D(32, (3, 3), padding='same', activation='elu')(uc3)
    
    
    oc = Conv2D(32, (3, 3), padding='same', activation='elu')(uc3)
    oc = Conv2D(64, (3, 3), padding='same', activation='elu')(oc)
    oc = Conv2D(32, (3, 3), padding='same', activation='elu')(oc)
    
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(oc)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model



if __name__ == '__main__':
 
    model = build_model()
    
    # add optimiser
    from custom_optimisers import NoisyAdam, NoisySGD
    model.compile(optimizer=NoisyAdam(), loss='binary_crossentropy')    
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy')    
    
    train = True
    
    import os
    if os.path.exists('./model/model-schlerp-{}.h5'.format(model_version)):
        print('loading saved model wights...')
        #train = False
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
    from data_utils import get_test_data
    Xtest = get_test_data()
    Xpreds = model.predict(Xtest)
    
    # view tests
    from skimage.io import imshow
    from matplotlib import pyplot as plt
    for img, mask in zip(Xtest, Xpreds):
        plt.subplot(1,2,1)
        img = img.reshape(IMG_HEIGHT, IMG_WIDTH)
        img = img
        imshow(img, cmap='Greens')
        plt.subplot(1,2,2)
        mask = mask.reshape(IMG_HEIGHT, IMG_WIDTH)
        mask = mask
        imshow(mask, cmap='Reds')
        plt.show()
