import glob
import os, datetime
import matplotlib.pyplot as plt
import numpy as np

import tensorboard
import tensorflow as tf
from tools import evaluate_model
from tensorflow.keras import mixed_precision
from TFRecordsFunctions import get_dataset, get_test_dataset
import tensorflow_io as tfio
from tensorflow.python.framework.ops import disable_eager_execution


from siamese_network import build_network, projection_layer, prediction_layer
from debugging import plot_sample_preprocessing, plot_sample_input_training, networkout_to_numpy, plot_std_mean_network_output, plot_PCA, plot_TSNE, plot_classes


def calculate_accuracy(y_pred, y_true):
    sum_all = np.dot(y_pred, y_true)
    print('Accuracy is {}'.format(sum_all/len(y_pred)))


def neg_cosine_similarity(p, z):

    p = tf.math.l2_normalize(p, axis=1)
    z = tf.stop_gradient(tf.math.l2_normalize(z, axis=1))

    product = tf.math.reduce_sum(tf.math.multiply(p, z), axis=1)

    return -tf.math.reduce_mean(product)


# def full_loss(pred1, pred2, output1, output2, class1, class2, y):
@tf.function
def full_loss(pred1, pred2, output1, output2):

    siamese_loss = tf.keras.losses.cosine_similarity(pred1, tf.stop_gradient(output2), axis=1) / 2 + tf.keras.losses.cosine_similarity(pred2, tf.stop_gradient(output1), axis=1) / 2

    #categorical_loss1 = tf.keras.losses.categorical_crossentropy(class1, y) / 2
    #categorical_loss2 = tf.keras.losses.categorical_crossentropy(class2, y) / 2

    #return -1 * siamese_loss + 0.5 * (categorical_loss1 + categorical_loss2)
    return siamese_loss


def full_loss_v2(pred1, pred2, output1, output2):

    return neg_cosine_similarity(pred1, output2) / 2 + neg_cosine_similarity(pred2, output1) / 2


@tf.function
def train_step1(x1, x2, loss):
    with tf.GradientTape() as tape:
        pred1, out1 = model(x1, training=True)
        pred2, out2 = model(x2, training=True)
        loss = loss(pred1, pred2, out1, out2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    std = 0.5 * tf.math.reduce_std(tf.math.l2_normalize(out1, axis=1), axis=1) + 0.5 * tf.math.reduce_std(tf.math.l2_normalize(out2, axis=1))

    return loss, std


class CustomModel(tf.keras.Model):
    def compile(self, optimizer, loss, metric):
        super().compile(optimizer)
        self.loss = loss
        self.metric = metric

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x1, x2, y = data

        loss, std = train_step1(x1, x2, self.loss)
        # Update metrics (includes the metric that tracks the loss)
        loss_tracker.update_state(loss)
        train_acc_metric.update_state(std)
        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result(), "std": train_acc_metric.result()}


if __name__ == "__main__":

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    audio_path = 'X:/sounds/VoxCeleb/vox1_dev_wav/wav/id10743/N94AAE1xEuE/00001.wav'
    model_name = 'SiameseNetwork_All_Losses'

    DEBUG = 0

    if DEBUG:
        plot_sample_preprocessing(audio_path)

    train_siamese = True

    # Settings for Extraction
    audio_length = 2.5
    window_length = 512
    hop_length = 512

    # Settings for Training
    EPOCHS = 100
    OUTPUT_DIMENSION = 2048

    loss_tracker = tf.keras.metrics.Mean(name="loss")
    train_acc_metric = tf.keras.metrics.Mean()

    if "scratch" in os.getcwd():
        tfrecord_files = glob.glob('/scratch/work/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        BATCH_SIZE = 128
        SHUFFLE_BUFFER_SIZE = 10
        #mixed_precision.set_global_policy('mixed_float16')

        print('Starting training on cluster')
    else:
        tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        tfrecord_files = tfrecord_files[:5]
        BATCH_SIZE = 24
        SHUFFLE_BUFFER_SIZE = 10
        #mixed_precision.set_global_policy('mixed_float16')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')
        print('Debugging Mode is turned on')
        modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
            "C:/Users/rechs1/PycharmProjects/SiamSpeak/modelcheckpoint",
            monitor="loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )

        # , profile_batch=(50, 100)

    if train_siamese:

        learning_rate = 0.05 * BATCH_SIZE / 256
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(learning_rate, 1000)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

        if DEBUG:
            tfrecord_files_test = glob.glob('C:/Users/rechs1/VoxCelebTFRecordAudioTest/*.tfrecords')
            tfrecord_files = tfrecord_files_test[:3]

        # Parameters for training the siamese network
        input_a = tf.keras.Input(shape=(96, 128, 1))

        # Get training data
        train_dataset = get_dataset(tfrecord_files, SHUFFLE_BUFFER_SIZE, BATCH_SIZE)

        if DEBUG:
            plot_sample_input_training(train_dataset)

        # Build network
        resnet_out = build_network(input_a, OUTPUT_DIMENSION)

        proj1 = projection_layer(resnet_out, OUTPUT_DIMENSION)
        pred1 = prediction_layer(proj1, OUTPUT_DIMENSION)

        print('-----------------------Training Starting---------------------')

        model = CustomModel(input_a, [pred1, proj1])

        model.compile(optimizer=optimizer, loss=full_loss, metric=train_acc_metric)

        if DEBUG:
            model.summary()

        if "scratch" not in os.getcwd():
            model.fit(x=train_dataset, epochs=EPOCHS, verbose=1, callbacks=[tensorboard_callback, modelcheckpoint])
        else:
            model.fit(x=train_dataset, epochs=EPOCHS, verbose=1)

        model.save(model_name)
        print('Model successfully saved under the name: {}'.format(model_name))
    #
    else:
        model = tf.keras.models.load_model('SiameseNetwork_All_Losses', custom_objects={'full_loss': full_loss})
        #model.summary()

    load_testSet = True

    # Load Test Set

    tfrecord_files_test = glob.glob('C:/Users/rechs1/VoxCelebTFRecordAudioTest/*.tfrecords')
    test_dataset, validation_dataset = get_test_dataset(tfrecord_files_test, model, SHUFFLE_BUFFER_SIZE=100, BATCHSIZE=50)

    if DEBUG:

        siamese_predictions, labels = networkout_to_numpy(test_dataset, model, OUTPUT_DIMENSION=2048)
        plot_std_mean_network_output(siamese_predictions, labels)
        plot_PCA(siamese_predictions, labels)
        plot_TSNE(siamese_predictions, labels)

    # Build and train classification layer
    model.trainable = False

    last_output = model.get_layer('OutputPrediction').output
    x = tf.keras.layers.Dense(512, activation='relu', name='Dense_Class1')(last_output)
    pred_class = tf.keras.layers.Dense(40, activation='softmax', name='Dense_Class')(x)

    classification_model = tf.keras.Model(model.input, pred_class)

    opt = tf.keras.optimizers.Adam(learning_rate=0.002)
    classification_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if DEBUG:
        classification_model.summary()

    classification_model.fit(test_dataset, shuffle=True, epochs=15, batch_size=50)

    prediction, label = evaluate_model(classification_model, validation_dataset)

    if DEBUG:
        plot_classes(prediction, label)

    x = 1
