import glob
import os, datetime
import matplotlib.pyplot as plt
import numpy as np
import siamese_network as sn
import tensorboard
import tensorflow as tf
from tensorflow.keras import mixed_precision
from TFRecordsFunctions import get_dataset, parse_tfr_element
from siamese_network import prediction_net


def neg_cosine_similarity(p, z):

    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

    product = tf.math.reduce_sum(tf.math.multiply(p, z), axis=1)

    return -tf.math.reduce_mean(product)


def full_loss(pred1, pred2, output1, output2):

    return neg_cosine_similarity(pred1, output2) / 2 + neg_cosine_similarity(pred2, output1) / 2


def linear_prediction(input_shape):
    input_a = tf.keras.Input(shape=input_shape)
    # MLP Projection Layer
    x = tf.keras.layers.Dense(512, activation='relu')(input_a)
    x = tf.keras.layers.BatchNormalization()(x)
    pred_class = tf.keras.layers.Dense(40, activation='softmax')(x)

    return tf.keras.Model(inputs=input_a, outputs=pred_class)


class CustomModel(tf.keras.Model):
    def compile(self, optimizer, my_loss):
        super().compile(optimizer)
        self.my_loss = my_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x1, x2, y = data

        with tf.GradientTape() as tape:
            out1 = self(x1, training=True)  # Forward pass
            out2 = self(x2, training=True)  # Forward pass

            pred1 = pred_net(out1, training=True)
            pred2 = pred_net(out2, training=True)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.my_loss(pred1, pred2, out1, out2)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        grads = self.optimizer.get_unscaled_gradients(gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}


if __name__ == "__main__":

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    DEBUG = 0
    load_model = False

    if DEBUG:
        print('Debugging Mode is turned on')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='30,50')
    else:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Settings for Extraction
    audio_length = 2.5
    window_length = 512
    hop_length = 512

    #test_database = ["X:/sounds/VoxCeleb/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav"]

    #tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebClean/*.tfrecords')

    if "scratch" in os.getcwd():
        tfrecord_files = glob.glob('/scratch/work/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        BATCH_SIZE = 128
        SHUFFLE_BUFFER_SIZE = 10
        mixed_precision.set_global_policy('mixed_float16')

        print('Starting training on cluster')
    else:
        tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        BATCH_SIZE = 24
        SHUFFLE_BUFFER_SIZE = 10
        mixed_precision.set_global_policy('mixed_float16')

    tfrecord_files = tfrecord_files[:5]
    #tfrecord_files = 'C:/Users/rechs1/VoxCelebClean/VoxCeleb_Set0.tfrecords'

    # Parameters for training the siamese network

    input_shape = (108, 128, 1)
    input_a = tf.keras.Input(shape=input_shape)

    # Get training data
    train_dataset = get_dataset(tfrecord_files).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    if DEBUG:
        test = train_dataset.take(1)
        for x1,x2, y in test.as_numpy_iterator():
            plt.imshow(x1[0])
            plt.show()
            print(x1.shape, y)

    # Build network
    proj_output = sn.build_network(input_a)
    pred_net = prediction_net((2048))

    print('-----------------------Training Starting---------------------')

    loss_tracker = tf.keras.metrics.Mean(name="loss")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model = CustomModel(input_a, proj_output)
    model.compile(optimizer=optimizer, my_loss=full_loss)

    if DEBUG:
        model.summary()

    model.fit(x=train_dataset, epochs=100, verbose='auto', callbacks=[tensorboard_callback])
    model.save('SiameseNetwork')

    #
    if load_model:
        model = tf.keras.models.load_model('SiameseNetwork', custom_objects={'full_loss': full_loss})
        model.summary()

    # Load Test Set
    tfrecord_files_test = glob.glob('C:/Users/rechs1/VoxCelebTFRecordTest/*.tfrecords')

    tfrecord_files_class_train = tfrecord_files_test[:int(0.8*tfrecord_files_test)]
    tfrecord_files_class_eval = tfrecord_files_test[int(0.8*tfrecord_files_test):]

    test_dataset = tf.data.TFRecordDataset(tfrecord_files_class_train)
    validation_dataset = tf.data.TFRecordDataset(tfrecord_files_class_eval)

    test_dataset = test_dataset.map(lambda x: parse_tfr_element(x, model)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)

    validation_dataset = validation_dataset.map(lambda x: parse_tfr_element(x, model)).shuffle(SHUFFLE_BUFFER_SIZE).batch(
        BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)

    # Build and train classification layer
    simple_predictor = linear_prediction(input_shape=(1, 512))

    simple_predictor.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=tf.keras.metrics.CategoricalAccuracy())

    if DEBUG:
        simple_predictor.summary()

    simple_predictor.fit(test_dataset, validation_data=validation_dataset, epochs=10)

    x = 1
