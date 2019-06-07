from keras.layers import Input
from keras.layers import (Conv2D, BatchNormalization, Activation, Add, Dense,
                          Flatten)
from keras.models import Model
from keras.optimizers import Adam


def EncoderBlock(input_layer, n_kernels, kernel_size, block_number):
    base_name = 'Encoder_{}_'.format(block_number)    
    
    # Convolution Branch
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(tmp_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_1')(tmp_layer)
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_2')(tmp_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same', strides=(2,2),
                       name=base_name + 'Conv_2')(tmp_layer)
    
    # Shortcut Branch
    input_layer = Conv2D(n_kernels, 1, padding='same',
                         strides=(2,2), name=base_name + 'Skip')(input_layer)
    
    # Merging Branches
    output_layer = Add(name=base_name + 'Add')([input_layer, tmp_layer])
    
    return output_layer


def build_model(n_kernels, kernel_size, dim=256):
    input_layer = Input(shape=(dim,dim,1), name='Input')

    tmp_layer = EncoderBlock(input_layer, n_kernels, kernel_size, 1)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 2)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 3)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 4)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 5)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 6)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 7)
    tmp_layer = EncoderBlock(tmp_layer, n_kernels, kernel_size, 8)
    
    output_layer = Dense(1, name='Output_Dense')(tmp_layer)
    output_layer = Flatten(name='Output_Flatten')(output_layer)
    model =  Model(input_layer, output_layer)
    optimizer = Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.summary()
    return model