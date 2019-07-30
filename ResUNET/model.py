import sys
# Hide backend message when importing keras
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
from keras.layers import (Conv2D, BatchNormalization, Activation, Add,
                          UpSampling2D, Concatenate, Input, Dropout)
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
sys.stderr = stderr


def dice_loss(y_true, y_pred):
    g = K.batch_flatten(y_true)
    p = K.batch_flatten(y_pred)
    epsilon = K.epsilon()
    p_dot_g = K.batch_dot(p,g, axes=1)
    p_dot_p = K.batch_dot(p,p, axes=1)
    g_dot_g = K.batch_dot(g,g, axes=1)
    DL = 1 - (2. * p_dot_g) / (p_dot_p + g_dot_g + epsilon)
    return K.transpose(K.mean(DL))


def EncoderBlock(input_layer,
                 n_kernels,
                 kernel_size,
                 block_number,
                 dropout):
    
    base_name = 'Encoder_{}_'.format(block_number)
    
    # Shortcut Branch
    shortcut_layer = Conv2D(n_kernels, 1, padding='same',
                            strides=(2,2), name=base_name + 'Skip')(input_layer)
    
    # Convolution Branch
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(input_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       strides=(2,2), name=base_name + 'Conv_1')(tmp_layer)
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_2')(tmp_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_2')(tmp_layer)
    
    # Merging Branches
    output_layer = Add(name=base_name + 'Add')([shortcut_layer, tmp_layer])
    output_layer = Dropout(dropout, name=base_name + 'Dropout')(output_layer)
    
    return output_layer


def BridgeBlock(input_layer,
                n_kernels,
                kernel_size,
                dropout):
    
    base_name = 'Bridge_'
    
    # Convolution Branch
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(input_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       strides=(2,2), name=base_name + 'Conv_1')(tmp_layer)
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
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
    output_layer = Dropout(dropout, name=base_name + 'Dropout')(output_layer)
    
    return output_layer


def DecoderBlock(input_layer,
                 carry_layer,
                 n_kernels,
                 kernel_size,
                 block_number,
                 dropout):
    
    base_name = 'Decoder_{}_'.format(block_number)
    
    # Merging inputs and carries
    input_layer = Concatenate(name=base_name + 'Concatenate')([input_layer,
                                                               carry_layer])
    
    # Convolution Branch
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_1')(input_layer)
    tmp_layer = Activation('relu', name=base_name + 'ReLU_1')(input_layer)
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name=base_name + 'Conv_1')(tmp_layer)
    tmp_layer = BatchNormalization(name=base_name + 'BatchNorm_2')(tmp_layer)
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
    output_layer = Dropout(dropout, name=base_name + 'Dropout')(output_layer)
    return output_layer


def ResUNET(n_kernels,
            kernel_size,
            depth,
            learn_rate,
            dropout, dim=256,
            output_length=1):
    
    input_layer = Input(shape=(dim,dim,1), name='Input')
    tmp_layer = Conv2D(n_kernels, kernel_size, padding='same',
                       name='Input_Conv')(input_layer)
    output_layer = ResUNET_helper(tmp_layer, n_kernels, kernel_size, depth,
                                  dropout)
    output_layer = Concatenate(name='Output_Concatenate')([input_layer,
                                                           output_layer])
    output_layer = Conv2D(n_kernels, kernel_size, padding='same',
                          name='Output_Conv_1')(output_layer)    
    output_layer = Conv2D(output_length, 1, padding='same',
                          name='Output_Conv_2')(output_layer)
    output_layer = Activation('sigmoid', name='Sigmoid')(output_layer)
    
    model = Model(input_layer, output_layer)
    optimizer = Adam(lr=learn_rate)
    model.compile(loss=dice_loss, optimizer=optimizer)
    model.summary()

    return model


def ResUNET_helper(input_layer,
                   n_kernels,
                   kernel_size,
                   depth,
                   dropout):
    
    if depth == 0:
        return BridgeBlock(input_layer, n_kernels, kernel_size, dropout)
    
    encoder = EncoderBlock(input_layer, n_kernels, kernel_size, depth, dropout)
    tmp_layer = ResUNET_helper(encoder, n_kernels*2, kernel_size, depth-1,
                               dropout)
    decoder = DecoderBlock(tmp_layer, encoder, n_kernels, kernel_size, depth,
                           dropout)
    
    return decoder