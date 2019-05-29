from keras.layers import Input
from keras.layers import Dropout, Concatenate
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2  
from keras import backend as K
import tensorflow as tf
from keras.backend.common import epsilon
from keras.callbacks import Callback

_weight=49


def get_weight():
    return _weight


class update_weight_callback(Callback):
    
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        global _weight
        logs = logs or {}
        
        _weight = max(1, 49-(49/400)*epoch)
        print('Updating weight: ', _weight)

        return


def weighted_cross_entropy_backend(target, output, weight):
    # convert output to logits
    _epsilon = tf.convert_to_tensor(epsilon(), dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    
    return tf.nn.weighted_cross_entropy_with_logits(target, output, weight)

    
def weighted_cross_entropy(y_true, y_pred, weight=_weight):
    # higher weight = more emphasis placed on 1's in y_true
    return K.mean(weighted_cross_entropy_backend(y_true, y_pred, weight), axis=-1)


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
    #model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.compile(loss=weighted_cross_entropy, optimizer = optimizer)
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