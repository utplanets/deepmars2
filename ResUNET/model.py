from keras.layers import (Conv2D, BatchNormalization, Activation, Add,
                          UpSampling2D, Concatenate, Input)
from keras.models import Model
from keras.optimizers import Adam


def EncoderBlock(input_layer, n_kernels, kernel_size, block_number):
    base_name = 'Encoder_{}_'.format(block_number)
    
    # Shortcut Branch
    shortcut_layer = Conv2D(n_kernels, 1, padding='same',
                            strides=(2,2), name=base_name + 'Skip')(input_layer)
    
    # Convolution Branch
    #tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(input_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       strides=(2,2), name=base_name + 'Conv_1')(tmp_layer)
    #tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_2')(tmp_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_2')(tmp_layer)
    
    # Merging Branches
    output_layer = Add(name=base_name + 'Add')([shortcut_layer, tmp_layer])
    
    return output_layer


def BridgeBlock(input_layer, n_kernels, kernel_size):
    base_name = 'Bridge_'
    
    # Convolution Branch
    #tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(input_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       strides=(2,2), name=base_name + 'Conv_1')(tmp_layer)
    #tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_2')(tmp_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_2')(tmp_layer)
    
    # Shortcut Branch
    input_layer = Conv2D(n_kernels, 1, padding='same',
                         strides=(2,2), name=base_name + 'Skip')(input_layer)
    
    # Merging Branches
    output_layer = Add(name=base_name + 'Add')([input_layer, tmp_layer])
    output_layer = UpSampling2D(size=(2,2),
                                name=base_name + 'UpSampling')(output_layer)
    
    return output_layer


def DecoderBlock(input_layer, carry_layer, n_kernels, kernel_size,
                 block_number):
    base_name = 'Decoder_{}_'.format(block_number)
    
    # Merging inputs and carries
    input_layer = Concatenate()([input_layer, carry_layer])
    
    # Convolution Branch
    #tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(input_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_1')(tmp_layer)
    #tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_2')(tmp_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_2')(tmp_layer)
    
    # Shortcut Branch
    input_layer = Conv2D(n_kernels, 1, padding='same',
                         name=base_name + 'Skip')(input_layer)
    
    # Merging Branches
    output_layer = Add(name=base_name + 'Add')([input_layer, tmp_layer])
    output_layer = UpSampling2D(size=(2,2),
                                name=base_name + 'UpSampling')(output_layer)
    
    return output_layer


def ResUNET(n_kernels, kernel_size, depth, learn_rate, dim=256):
    input_layer = Input(shape=(dim,dim,1), name='Input')
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name='Input_Conv')(input_layer)
    output_layer = ResUNET_helper(tmp_layer, n_kernels, kernel_size, depth)    
    output_layer = Conv2D(1, 1, padding='same',
                          name='Output_Conv')(output_layer)
    output_layer = Activation('sigmoid', name='Sigmoid')(output_layer)
    
    model = Model(input_layer, output_layer)
    optimizer = Adam(lr=learn_rate)
    #model.compile(loss='mean_squared_error', optimizer = optimizer)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.summary()
 #   from keras.utils import plot_model
 #   plot_model(model, to_file='model.png')
    return model


def ResUNET_helper(input_layer, n_kernels, kernel_size, depth):
    if depth == 0:
        return BridgeBlock(input_layer, n_kernels, kernel_size)
    
    encoder = EncoderBlock(input_layer, n_kernels, kernel_size, depth)
    tmp_layer = ResUNET_helper(encoder, n_kernels*2, kernel_size, depth-1)
    decoder = DecoderBlock(tmp_layer, encoder, n_kernels, kernel_size, depth)
    
    return decoder