import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv1D, Normalization, Conv2D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop, Adamax, AdamW, Adam, SGD


def create_1d_cnn(optimizer, activ, bias):
    #filters = [32,64,96,96,128,128]
    #filters = [16,32,64,96,128,128]
    kernels = [12,11,11,12,11,11]
    strides = [3,4,4,5,5,5]
    filters = [16,32,64,96,128,128]
    dropouts = [0.,0.]
    #dropouts = [0.2,0.2]
    #dropouts = [0.5,0.4]
    model = Sequential()
    # input batch_size x 27090 x 1channel
    model.add(Conv1D(filters[0], kernel_size=12, strides=3, padding="valid",
                    use_bias=bias,
              activation=activ, input_shape=(27090,1)))
    # output shape= batch_size x 9027 x 64 channels
    model.add(BatchNormalization())
    model.add(Conv1D(filters[1], kernel_size=11, strides=4, padding="valid",
                     use_bias=bias,
              activation=activ)) # hard_sigmoid
    # output shape = batch_size x 2255 x 64
    model.add(BatchNormalization())
    model.add(Conv1D(filters[2], kernel_size=11, strides=4,
                     activation=activ,use_bias=bias))
    # output shape = batch_size x 562 x 96
    model.add(BatchNormalization())
    model.add(Conv1D(filters[3], kernel_size=12, strides=5,
                     activation=activ,use_bias=bias))
    # output shape = batch_size x 111 x 96 ==>
    model.add(BatchNormalization())
    model.add(Conv1D(filters[4], kernel_size=11, strides=5,
                     activation=activ,use_bias=bias))
    # output shape = batch_size x 21 x 128 ==> 2688
    model.add(BatchNormalization())

    model.add(Conv1D(filters[5], kernel_size=11, strides=5,
                     activation=activ,use_bias=bias))
    model.add(BatchNormalization())
    # output shape = batch_size x 3 x 128 ==> 384
    #model.add(BatchNormalization())

    model.add(Flatten())
    # output shape = batch_size x 384
    d_size = int(filters[5]*3)
    model.add(Dense(d_size, activation="relu"))
    model.add(Dropout(rate=dropouts[0]))
    model.add(Dense(128, activation="relu"))
    #model.add(Dense(int(d_size/2), activation="relu"))
    model.add(Dropout(rate=dropouts[1]))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_2d_cnn(optimizer, activ):
    model = Sequential()
    # input batch x 105x258x1
    model.add(Conv2D(16, kernel_size=4, strides=2,
                     padding="valid", activation="softsign",
                     input_shape=(105,258,1)))
    model.add(BatchNormalization())
    # output b x 51 x 128 x 16 ch
    model.add(Conv2D(24, kernel_size=4, strides=2,padding="valid",
                     activation="softsign"))
    model.add(BatchNormalization())
    # output b x 24 x 63 x 24
    model.add(Conv2D(32, kernel_size=4, strides=2,padding="valid",
                     activation="softsign"))
    model.add(BatchNormalization())
    # output b x 11 x 30 x 32
    model.add(Conv2D(48, kernel_size=4, strides=2,padding="valid",
                     activation="softsign"))
    model.add(BatchNormalization())
    # output b x 4 x 14 x 48
    model.add(Conv2D(64, kernel_size=4, strides=2,padding="valid",
                     activation="softsign"))
    model.add(BatchNormalization())
    # output b x 1 x 6 x 64
    model.add(Flatten())
    # output shape = batch_size x 384
    model.add(Dense(int(64*6), activation="relu"))
    model.add(Dropout(rate=0.4))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def create_mlp(optimizer, activ):
    model = Sequential()
    model.add(tf.keras.Input(shape=(27090,)))
    model.add(Dense(27090, activation=activ))
    model.add(Dropout(rate=0.2))
    model.add(Dense(10000, activation=activ))
    model.add(Dropout(rate=0.3))
    model.add(Dense(5000, activation=activ))
    model.add(Dropout(rate=0.3))
    model.add(Dense(2500, activation=activ))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1000, activation=activ))
    model.add(Dropout(rate=0.3))
    model.add(Dense(500, activation=activ))
    model.add(Dropout(rate=0.3))
    model.add(Dense(250, activation=activ))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model

