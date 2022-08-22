import tensorflow as tf


def res_block(x, filters, first_layer=False):

    # Perform the original mapping
    if first_layer:
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        x_skip = x
    else:
        x_skip = x
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add the skip connection to the regular mapping
    x = tf.keras.layers.Add()([x, x_skip])

    # Non linearly activate the result
    x = tf.keras.layers.Activation("relu")(x)

    # Return the result
    return x


def stacked_ResBlocks(x):
    # There is 6n layers and each group contains 2n layers.
    filters = 64
    for layer_group in range(4):

        # Each block in our code has 2 weighted layers,
        # and each group has 2n such blocks,
        # so 2n/2 = n blocs per group.

        for block in range(2):

            # Perform filter size increase at every
            # first layer in the 2nd block onwards.
            # Apply Conv block for projecting the skip
            # connection.
            if layer_group > 0 and block == 0:
                filters *= 2
                x = res_block(x, filters, first_layer=True)
            else:
                x = res_block(x, filters)

        # Return final layer
    return x


def backbone(input, dense_num):

    x = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same")(input)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = stacked_ResBlocks(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(dense_num, name="OutputEncoder")(x)
    output = tf.keras.layers.BatchNormalization()(output)

    return output


def build_network(input_a, dense_num):
    """"
    input_a: Input dimension for the first layer in the network
    dense_num: Output dimensionality for projection and prediction layer (and final output)

    returns last layer of the output
    """

    # Backbone is a ResNet50 architecture
    output = backbone(input_a, dense_num)

    return tf.keras.Model(input_a, output, name='backbone')


def projection_layer(input, output_dimension):
    x = tf.keras.layers.Dense(output_dimension//2, activation='relu', use_bias=False)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(output_dimension, use_bias=False)(x)
    return output


def prediction_layer(input, output_dimension):

    proj_output = tf.keras.layers.Dense(output_dimension // 4, use_bias=False)(input)
    prediction_layer = tf.keras.layers.BatchNormalization()(proj_output)
    prediction_layer = tf.keras.layers.Activation('relu')(prediction_layer)
    prediction_layer = tf.keras.layers.Dense(output_dimension)(prediction_layer)

    return prediction_layer
