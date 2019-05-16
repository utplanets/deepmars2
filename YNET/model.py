from keras.layers import Input
from keras.layers import Dropout, Concatenate
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2  


def build_YNET(depth, n_inputs, dim, kernel_size, n_kernels,
               dropout_rate, learn_rate, lmbda=0.01,
               activation_function='softplus'):
    
    inputs = [Input(shape=(dim, dim, 1)) for _ in range(n_inputs)]
    
    output = YNET_helper(depth, inputs, kernel_size, n_kernels,
                         dropout_rate, lmbda, activation_function)
    
    output = Conv2D(1, 1, activation='sigmoid', padding='same',
                    kernel_regularizer=l2(lmbda))(output)
    
    model = Model(inputs, output)
    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.summary()

    return model


def YNET_helper(depth, inputs, kernel_size, n_kernels, dropout_rate, lmbda,
                activation_function):
    
    if depth == 1:
        if len(inputs) == 1:
            return inputs[0]
        else:
            inputs = Concatenate(axis=3)(inputs)
            outputs = SeparableConv2D(n_kernels, kernel_size,
                                      activation=activation_function,
                                      padding='same',
                                      kernel_regularizer=l2(lmbda))(inputs)
            outputs = SeparableConv2D(n_kernels, kernel_size,
                                      activation=activation_function,
                                      padding='same',
                                      kernel_regularizer=l2(lmbda))(outputs)
            return outputs
    
    inputs = [Conv2D(n_kernels, kernel_size, activation=activation_function,
                     padding='same', kernel_regularizer=l2(lmbda))(layer)
              for layer in inputs]
    inputs = [Conv2D(n_kernels, kernel_size, activation=activation_function,
                     padding='same', kernel_regularizer=l2(lmbda))(layer)
              for layer in inputs]
    pooled = [MaxPooling2D(pool_size=(2,2))(layer) for layer in inputs]
    
    outputs = YNET_helper(depth-1, pooled, kernel_size, n_kernels * 2,
                          dropout_rate, lmbda, activation_function)
    outputs = UpSampling2D(size=(2,2))(outputs)
    outputs = Concatenate(axis=3)(inputs + [outputs])
    outputs = Dropout(dropout_rate)(outputs)
    outputs = SeparableConv2D(n_kernels, kernel_size,
                     activation=activation_function,
                     padding='same', kernel_regularizer=l2(lmbda))(outputs)
    outputs = SeparableConv2D(n_kernels, kernel_size,
                     activation=activation_function,
                     padding='same', kernel_regularizer=l2(lmbda))(outputs)
    return outputs