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

    return tf.keras.Model(inputs=input, outputs=output)


def prediction_net(input):
    prediction_layer = tf.keras.layers.Dense(512, activation='relu')(input)
    prediction_layer = tf.keras.layers.BatchNormalization()(prediction_layer)
    prediction_layer = tf.keras.layers.Dense(2048)(prediction_layer)

    return tf.keras.Model(inputs=input, outputs=prediction_layer)


def build_network(input_shape):

    # Linear Projection MLP
    input_one = tf.keras.Input(shape=input_shape)
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)
    input_pred = tf.keras.Input(shape=(2048, ))

    backbone_net = backbone(input_one)

    #backbone_net.summary()

    output1 = backbone_net(input_a)
    output2 = backbone_net(input_b)

    pred_net = prediction_net(input_pred)
    #pred_net.summary()

    pred1 = pred_net(output1)
    pred2 = pred_net(output2)

    return tf.keras.Model([input_a, input_b], [output1, output2, pred1, pred2])


