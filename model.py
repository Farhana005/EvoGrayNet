import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, SeparableConv2D, Lambda, Multiply, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Activation, Add


def dilated_feature_extractor(inputs, rates, dropout_rate):   #dilated_feature_extractor
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


def feature_recalibration(inputs, ratio):    #feature_recalibration 
    filters = inputs.shape[-1]

    se = GlobalAveragePooling2D()(inputs)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)

    return Multiply()([inputs, se])

def gray_module(inputs, filters, dropout_rate):
    # Initial 3x3 convolution with BatchNorm, ReLU activation, and L2 regularization
    
    regularization_factor=1e-4
    
    conv1 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(regularization_factor))(inputs)
    batch1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.nn.relu(batch1)
    act1 = Dropout(dropout_rate)(act1)

    # Depthwise separable convolution with BatchNorm, ReLU, and L2 regularization
    conv_dw = DepthwiseConv2D(3, padding='same', depth_multiplier=1,
                              kernel_initializer='he_normal',
                              depthwise_regularizer=l2(regularization_factor))(act1)
    batch_dw = tf.keras.layers.BatchNormalization()(conv_dw)
    act_dw = tf.nn.relu(batch_dw)
    act_dw = Dropout(dropout_rate)(act_dw)

    # Pointwise convolutions with BatchNorm, ReLU, and L2 regularization
    conv_pw1 = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(regularization_factor))(act_dw)
    batch_pw1 = tf.keras.layers.BatchNormalization()(conv_pw1)
    act_pw1 = tf.nn.relu(batch_pw1)
    act_pw1 = Dropout(dropout_rate)(act_pw1)

    conv_pw2 = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(regularization_factor))(act_pw1)

    return Concatenate(axis=3)([act1, conv_pw2])




# # Define the attention gate function
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


# Update the build_model function to include attention gates
def build_model(random_architecture, input_shape=(256, 256, 3)):
    gray_filters = random_architecture['gray_filters']
    dilated_feature_extractor_rates = random_architecture['dilated_feature_extractor_rates']
    feature_recalibration_ratio = random_architecture['feature_recalibration_ratio']
    # network_depth = random_architecture['network_depth']
    dropout_rate = random_architecture['dropout_rate']
    final_activation = random_architecture["final_activation"]

    # Ensure dilated_feature_extractor_rates only contains positive values
    dilated_feature_extractor_rates = [rate for rate in dilated_feature_extractor_rates if rate > 0]

    inputs = layers.Input(shape=input_shape)

    # Encoder path with gray modules and feature_recalibration
    x1 = gray_module(inputs, filters=gray_filters, dropout_rate=dropout_rate)
    x1 = feature_recalibration(x1, ratio=feature_recalibration_ratio)

    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = gray_module(pool1, filters=gray_filters, dropout_rate=dropout_rate)
    x2 = feature_recalibration(x2, ratio=feature_recalibration_ratio)

    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = gray_module(pool2, filters=gray_filters, dropout_rate=dropout_rate)
    x3 = feature_recalibration(x3, ratio=feature_recalibration_ratio)

    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = gray_module(pool3, filters=gray_filters, dropout_rate=dropout_rate)
    x4 = feature_recalibration(x4, ratio=feature_recalibration_ratio)

    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = gray_module(pool4, filters=gray_filters, dropout_rate=dropout_rate)
    x5 = feature_recalibration(x5, ratio=feature_recalibration_ratio)

    # dilated_feature_extractor block
    x5 = dilated_feature_extractor(x5, rates=dilated_feature_extractor_rates, dropout_rate=dropout_rate)

    # Decoder path with upsampling, attention, and concatenation
    up5 = UpSampling2D(size=(2, 2))(x5)
    attn4 = attention_gate(x4, up5, inter_channels=gray_filters // 2)
    up5 = Concatenate()([up5, attn4])

    up4 = UpSampling2D(size=(2, 2))(up5)
    attn3 = attention_gate(x3, up4, inter_channels=gray_filters // 2)
    up4 = Concatenate()([up4, attn3])

    up3 = UpSampling2D(size=(2, 2))(up4)
    attn2 = attention_gate(x2, up3, inter_channels=gray_filters // 2)
    up3 = Concatenate()([up3, attn2])

    up2 = UpSampling2D(size=(2, 2))(up3)
    attn1 = attention_gate(x1, up2, inter_channels=gray_filters // 2)
    up2 = Concatenate()([up2, attn1])

    # Final convolution and output layer
    all_conv = gray_module(up2, filters=gray_filters, dropout_rate=dropout_rate)
    all_conv = dilated_feature_extractor(all_conv, rates=dilated_feature_extractor_rates, dropout_rate=dropout_rate)
    final_conv = Conv2D(1, 1, activation=final_activation)(all_conv)

    model = tf.keras.Model(inputs=inputs, outputs=final_conv)
    return model

