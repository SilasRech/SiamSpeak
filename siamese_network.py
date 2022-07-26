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


def backbone(input):

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = stacked_ResBlocks(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    # MLP Projection Layer
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048)(x)
    output = tf.keras.layers.BatchNormalization()(x)

    return output


def prediction_net(input):
    input_a = tf.keras.Input(shape=input)
    prediction_layer = tf.keras.layers.Dense(2048, activation='relu')(input_a)
    prediction_layer = tf.keras.layers.BatchNormalization()(prediction_layer)
    prediction_layer = tf.keras.layers.Dense(2048, name="OutputPrediction", dtype='float32')(prediction_layer)

    return tf.keras.Model(input_a, prediction_layer)


def projection_net(input):

    prediction_layer1 = tf.keras.layers.Dense(2048, activation='relu')(input)
    prediction_layer1 = tf.keras.layers.BatchNormalization()(prediction_layer1)
    pred1 = tf.keras.layers.Dense(2048, name="OutputPrediction1", dtype='float32')(prediction_layer1)

    return pred1


def build_network(input_a):

    output1 = backbone(input_a)
    proj_output = projection_net(output1)

    return proj_output




