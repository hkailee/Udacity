from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, Reshape, MaxPooling1D, Flatten)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simple_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model



def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, pooling=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
        pooling (int): Maxpooling size
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // (stride * pooling)


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    # Adding 1st layer
    activation='relu'
    for layer in range(recur_layers):
        if layer == 0:
            simple_rnn = GRU(units, activation=activation,
                             return_sequences=True, implementation=2, name='rnn'+str(layer))(input_data)
            bn_rnn = BatchNormalization()(simple_rnn)
        else:
            simple_rnn = GRU(units, activation=activation,
                             return_sequences=True, implementation=2, name='rnn'+str(layer))(bn_rnn)
            bn_rnn = BatchNormalization()(simple_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    activation='relu'
    bidir_rnn = Bidirectional(GRU(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn'),merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)     
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim=161, 
                filters=200,
                kernel_size=3,
                conv_stride=1,
                conv_border_mode='same',
                units=200, 
                cnn_layers=3,
                recur_layers=3, 
                layer_dropout_rate=0.3, 
                recurrent_dropout_rate=0.3, 
                maxpool_size=2,
                output_dim=29):                    
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # TODO: Specify the layers in your network
    activation='relu'
    
    """
    # Add 2D convolutional layer (Failed - to come back later after graduation)
    input_shape = (13, 13, 1)
    
    conv_2d_1 = Conv2D(filters, kernel_size, 
                     strides=conv_stride, 
                     input_shape=input_shape,
                     padding=conv_border_mode,
                     activation=activation,
                     name='conv2d')(input_data)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(Reshape(input_shape)(onv_2d_1)
    
    # Add batch normalization and 
    bn_cnn = BatchNormalization(name='bn_pool_1')(pool_1)
    reshaped_bn_cnn = Reshape((36, 200))(bn_cnn)
    """
    
    # Add convolutional layer
    temp_layer = input_data
    
    for i in range(cnn_layers):
        # Add convolutional layer
        conv_1d = Conv1D(filters, 
                     kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation=activation,
                     name=f'conv1d_{i+1}')(temp_layer)

        # Add batch normalization
        bn_cnn = BatchNormalization(name=f'bn_conv_1d_{i+1}')(conv_1d)
        
        # Add dropout after convolution
        if layer_dropout_rate > 0:
            temp_layer = Dropout(layer_dropout_rate, name=f'cnn_dropout_{i+1}')(bn_cnn)
        else:
            temp_layer = bn_cnn
    
    
    # Add max pooling 
    pool_cnn = MaxPooling1D(maxpool_size, name=f'cnn_max_pool_{i+1}')(temp_layer)
    
    temp_layer = pool_cnn
    for i in range(recur_layers):
        simple_rnn = Bidirectional(GRU(units, activation=activation,
                                           return_sequences=True, implementation=2, 
                                           name='rnn'+str(i), 
                                           dropout_U=recurrent_dropout_rate), 
                                           merge_mode='concat')(temp_layer)
        bn_rnn = BatchNormalization()(simple_rnn)
        # Add dropout after convolution
        if layer_dropout_rate > 0:
            temp_layer = Dropout(layer_dropout_rate, name=f'bn_rnn_dropout_{i+1}')(bn_rnn)
        else:
            temp_layer = bn_rnn
                    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(temp_layer)        
            
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax',name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
                                   x, kernel_size, conv_border_mode, conv_stride, pooling=maxpool_size)

    print(model.summary())
    return model