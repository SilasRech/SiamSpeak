import tensorflow as tf


def res_block(x, filters, first_layer=False):
    # Create skip connection
    x_skip = x

    # Perform the original mapping
    if first_layer:
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding="same")(x_skip)
    else:
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x_skip)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # Perform matching of filter numbers if necessary
    if first_layer:
        x_skip = tf.keras.layers.Lambda(lambda x: tf.pad(x[:, ::2, ::2, :], tf.constant(
            [[0, 0, ], [0, 0], [0, 0], [filters // 4, filters // 4]]), mode="CONSTANT"))(x_skip)

    # Add the skip connection to the regular mapping
    x = tf.keras.layers.Add()([x, x_skip])

    # Nonlinearly activate the result
    x = tf.keras.layers.Activation("relu")(x)

    # Return the result
    return x


def stacked_ResBlocks(x):
    # There is 6n layers and each group contains 2n layers.
    filters = 64
    for layer_group in range(3):

        # Each block in our code has 2 weighted layers,
        # and each group has 2n such blocks,
        # so 2n/2 = n blocks per group.
        for block in range(3):

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

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(input)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = stacked_ResBlocks(x)
    output = tf.keras.layers.GlobalAveragePooling2D()(x)

    return output


def build_network(input_a, dense_num):
    """"
    input_a: Input dimension for the first layer in the network
    dense_num: Output dimensionality for projection and prediction layer (and final output)

    returns last layer of the output
    """

    # Backbone is a ResNet50 architecture
    output1 = backbone(input_a, dense_num)

    return output1


def projection_layer(input, output_dimension):
    x = tf.keras.layers.Dense(output_dimension, use_bias=False)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(output_dimension)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(output_dimension)(x)
    output = tf.keras.layers.BatchNormalization()(x)

    return output


def prediction_layer(input, output_dimension):

    proj1 = tf.keras.layers.Dense(output_dimension, name="OutputProjection", use_bias=False)(input)

    proj_output = tf.keras.layers.Dense(output_dimension // 4)(proj1)
    prediction_layer = tf.keras.layers.BatchNormalization()(proj_output)
    prediction_layer = tf.keras.layers.Activation('relu')(prediction_layer)
    prediction_layer = tf.keras.layers.Dense(output_dimension, name="OutputPrediction")(prediction_layer)

    return prediction_layer
