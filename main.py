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


def calculate_accuracy(y_pred, y_true):
    sum_all = np.dot(y_pred, y_true)
    print('Accuracy is {}'.format(sum_all/len(y_pred)))


def neg_cosine_similarity(p, z):

    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

    product = tf.math.reduce_sum(tf.math.multiply(p, z), axis=1)

    return -tf.math.reduce_mean(product)


@tf.function(jit_compile=True)
def full_loss(pred1, pred2, output1, output2):

    return tf.keras.losses.cosine_similarity(pred1, output2, axis=1) / 2 + tf.keras.losses.cosine_similarity(pred2, output1, axis=1) / 2


def full_loss_v2(pred1, pred2, output1, output2):

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
            #scaled_loss = self.optimizer.get_scaled_loss(loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        #grads = self.optimizer.get_unscaled_gradients(gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}


if __name__ == "__main__":

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    DEBUG = 1
    train_siamese = True
    # Settings for Extraction
    audio_length = 2.5
    window_length = 512
    hop_length = 512

    # Settings for Training
    EPOCHS = 100

    #test_database = ["X:/sounds/VoxCeleb/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav"]

    #tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebClean/*.tfrecords')

    if "scratch" in os.getcwd():
        tfrecord_files = glob.glob('/scratch/work/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        BATCH_SIZE = 256
        SHUFFLE_BUFFER_SIZE = 10
        mixed_precision.set_global_policy('mixed_float16')

        print('Starting training on cluster')
    else:
        tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        BATCH_SIZE = 20
        SHUFFLE_BUFFER_SIZE = 10
        #mixed_precision.set_global_policy('mixed_float16')

        if DEBUG:
            print('Debugging Mode is turned on')
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                'SiameseNetwork',
                monitor="loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
            )

        else:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if train_siamese:
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

        if "scratch" not in os.getcwd():
            model.fit(x=train_dataset, epochs=EPOCHS, verbose='auto', callbacks=[tensorboard_callback, modelcheckpoint_callback])
        else:
            model.fit(x=train_dataset, epochs=EPOCHS, verbose='auto')

        model.save('SiameseNetwork')

    #
    else:
        model = tf.keras.models.load_model('SiameseNetwork', custom_objects={'full_loss': full_loss})
        #model.summary()

    load_testSet = True

    # Load Test Set
    if load_testSet:
        tfrecord_files_test = glob.glob('C:/Users/rechs1/VoxCelebTFRecordAudioTest/*.tfrecords')

        tfrecord_files_class_train = tfrecord_files_test[:3]
        tfrecord_files_class_eval = tfrecord_files_test[-1]

        test_dataset = tf.data.TFRecordDataset(tfrecord_files_class_train)
        validation_dataset = tf.data.TFRecordDataset(tfrecord_files_class_eval)

        test_dataset = test_dataset.map(lambda x: parse_tfr_element(x, phase='eval', model=model)).shuffle(SHUFFLE_BUFFER_SIZE).batch(1).prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.map(lambda x: parse_tfr_element(x, phase='eval', model=model)).shuffle(
            SHUFFLE_BUFFER_SIZE).batch(1).prefetch(
            tf.data.AUTOTUNE)

        validation_dataset_prediction = []
        test_dataset_prediction = []
        test_dataset_numpy_labels = []
        validation_dataset_numpy_labels = []
        for data, labels in test_dataset:
            test_dataset_numpy_labels.append(labels.numpy())
            test_dataset_prediction.append(model.predict(data, verbose=0))

        test_dataset_numpy_labels = np.array(test_dataset_numpy_labels)
        test_dataset_prediction = np.array(test_dataset_prediction)

        for data, labels in validation_dataset:
            validation_dataset_numpy_labels.append(labels.numpy())
            validation_dataset_prediction.append(model.predict(data, verbose=0))

        validation_dataset_numpy_labels = np.array(validation_dataset_numpy_labels)
        validation_dataset_prediction = np.array(validation_dataset_prediction)

        np.savez('C:/Users/rechs1/VoxCelebTFRecordAudioTest/TestClassifier.npz', x_train=test_dataset_prediction, y_train=test_dataset_numpy_labels, x_eval=validation_dataset_prediction, y_eval=validation_dataset_numpy_labels)
    else:

        test_zip = np.load('C:/Users/rechs1/VoxCelebTFRecordAudioTest/TestClassifier.npz')

        test_dataset_prediction = test_zip['x_train']
        test_dataset_numpy_labels = test_zip['y_train']
        validation_dataset_numpy_labels = test_zip['y_eval']
        validation_dataset_prediction = test_zip['x_eval']

    # Build and train classification layer
    simple_predictor = linear_prediction(input_shape=(1, 2048))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    simple_predictor.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if DEBUG:
        simple_predictor.summary()

    simple_predictor.fit(test_dataset_prediction, test_dataset_numpy_labels, validation_data=[validation_dataset_prediction, validation_dataset_numpy_labels], shuffle=True, epochs=10, batch_size=50)

    accuracy_test = simple_predictor.predict(validation_dataset_prediction)
    accuracy_test = np.squeeze(accuracy_test)
    new_mat = np.zeros(accuracy_test.shape)  # our zeros and ones will go here

    y_pred = np.argmax(np.squeeze(validation_dataset_numpy_labels), axis=1)
    accuracy_test1 = np.argmax(accuracy_test, axis=1)
    accuracy = np.mean(y_pred == accuracy_test1)
    print('Classfication Accuracy is {}'.format(accuracy))

    x = 1
