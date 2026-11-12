import tensorflow as tf
from tensorflow.keras.layers import (DepthwiseConv2D, Input, Conv2D, MaxPooling2D,
                                     UpSampling2D, Concatenate, SeparableConv2D,
                                     Lambda, Multiply, GlobalAveragePooling2D,
                                     Reshape, Dense, Dropout)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Activation, Add



# Define the Gray Module
def gray_module(inputs, filters, dropout_rate):
    # Initial 3x3 convolution with BatchNorm, ReLU activation, and L2 regularization
    regularization_factor = 1e-4

    conv1 = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(regularization_factor))(inputs)
    batch1 = layers.BatchNormalization()(conv1)
    act1 = layers.ReLU()(batch1)
    act1 = Dropout(dropout_rate)(act1)

    # Depthwise separable convolution with BatchNorm, ReLU, and L2 regularization
    conv_dw = DepthwiseConv2D(3, padding='same', depth_multiplier=1,
                               depthwise_initializer='he_normal',  # Changed here
                               depthwise_regularizer=l2(regularization_factor))(act1)
    batch_dw = layers.BatchNormalization()(conv_dw)
    act_dw = layers.ReLU()(batch_dw)
    act_dw = Dropout(dropout_rate)(act_dw)

    # Pointwise convolutions with BatchNorm, ReLU, and L2 regularization
    conv_pw1 = layers.Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(regularization_factor))(act_dw)
    batch_pw1 = layers.BatchNormalization()(conv_pw1)
    act_pw1 = layers.ReLU()(batch_pw1)
    act_pw1 = Dropout(dropout_rate)(act_pw1)

    conv_pw2 = layers.Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(regularization_factor))(act_pw1)

    return Concatenate(axis=3)([act1, conv_pw2])


# Define the ASPP Block
def aspp_block(inputs, rates, dropout_rate):
    filters = inputs.shape[-1]

    aspp1 = Conv2D(filters, 3, dilation_rate=rates[0], padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    aspp1 = Dropout(dropout_rate)(aspp1)  # Dropout after the first dilation convolution

    aspp2 = Conv2D(filters, 3, dilation_rate=rates[1], padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    aspp2 = Dropout(dropout_rate)(aspp2)  # Dropout after the second dilation convolution

    aspp3 = Conv2D(filters, 3, dilation_rate=rates[2], padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    aspp3 = Dropout(dropout_rate)(aspp3)  # Dropout after the third dilation convolution

    global_avg_pool = GlobalAveragePooling2D()(inputs)
    global_avg_pool = Reshape((1, 1, filters))(global_avg_pool)
    global_avg_pool = Conv2D(filters, 1, activation='relu', kernel_initializer='he_normal')(global_avg_pool)
    global_avg_pool = UpSampling2D(size=(inputs.shape[1], inputs.shape[2]))(global_avg_pool)

    concatenated = Concatenate(axis=3)([aspp1, aspp2, aspp3, global_avg_pool])
    result = Conv2D(filters, 1, activation='relu', kernel_initializer='he_normal')(concatenated)

    return result

# Define the Squeeze and Excitation Block
def squeeze_excitation(inputs, ratio):
    filters = inputs.shape[-1]

    se = GlobalAveragePooling2D()(inputs)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)

    return Multiply()([inputs, se])
# Define the attention gate function
def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, 1, padding='same', kernel_initializer='he_normal')(x)
    phi_g = Conv2D(inter_channels, 1, padding='same', kernel_initializer='he_normal')(g)

    # Element-wise addition of theta_x and phi_g followed by ReLU activation
    add = Add()([theta_x, phi_g])
    relu = Activation('relu')(add)

    psi = Conv2D(1, 1, padding='same', kernel_initializer='he_normal')(relu)
    sigmoid = Activation('sigmoid')(psi)

    # Apply the attention map to the input feature map x
    attn_out = Multiply()([x, sigmoid])
    return attn_out


from tensorflow.keras import layers, Model

def build_model(random_architecture, input_shape=(256, 256, 3)):
    gray_filters = random_architecture['gray_filters']
    aspp_dilation_rates = random_architecture['aspp_dilation_rates']
    squeeze_excitation_ratio = random_architecture['squeeze_excitation_ratio']
    network_depth = architecture['network_depth']
    dropout_rate = random_architecture['dropout_rate']
    final_activation = random_architecture["final_activation"]

    # Ensure aspp_dilation_rates only contains positive values
    aspp_dilation_rates = [rate for rate in aspp_dilation_rates if rate > 0]

    inputs = layers.Input(shape=input_shape)

    x1 = gray_module(inputs, filters=gray_filters, dropout_rate=dropout_rate)
    x1 = squeeze_excitation(x1, ratio=squeeze_excitation_ratio)

    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = gray_module(pool1, filters=gray_filters, dropout_rate=dropout_rate)
    x2 = squeeze_excitation(x2, ratio=squeeze_excitation_ratio)

    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = gray_module(pool2, filters=gray_filters, dropout_rate=dropout_rate)
    x3 = squeeze_excitation(x3, ratio=squeeze_excitation_ratio)

    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = gray_module(pool3, filters=gray_filters, dropout_rate=dropout_rate)
    x4 = squeeze_excitation(x4, ratio=squeeze_excitation_ratio)

    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = gray_module(pool4, filters=gray_filters, dropout_rate=dropout_rate)
    x5 = squeeze_excitation(x5, ratio=squeeze_excitation_ratio)

    x5 = aspp_block(x5, rates=aspp_dilation_rates, dropout_rate=dropout_rate)

    up5 = UpSampling2D(size=(input_shape[0] // x5.shape[1], input_shape[1] // x5.shape[2]))(x5)
    up4 = UpSampling2D(size=(input_shape[0] // x4.shape[1], input_shape[1] // x4.shape[2]))(x4)
    up3 = UpSampling2D(size=(input_shape[0] // x3.shape[1], input_shape[1] // x3.shape[2]))(x3)
    up2 = UpSampling2D(size=(input_shape[0] // x2.shape[1], input_shape[1] // x2.shape[2]))(x2)

    up_scaled = Concatenate()([x1, up2, up3, up4, up5])
    all_conv = gray_module(up_scaled, filters=gray_filters, dropout_rate=dropout_rate)
    all_conv = aspp_block(all_conv, rates=aspp_dilation_rates, dropout_rate=dropout_rate)

    final_conv = Conv2D(1, 1, activation=final_activation)(all_conv)

    model = tf.keras.Model(inputs=inputs, outputs=final_conv)
    return model
