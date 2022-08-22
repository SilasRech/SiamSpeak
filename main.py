import glob
import os, datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import tensorboard
import tensorflow as tf
from tensorboard.plugins import projector
import tensorflow_addons as tfa
import tools
from tools import evaluate_model, get_input_shape
from tensorflow.keras import mixed_precision
from TFRecordsFunctions import get_dataset, get_test_dataset, get_id
import tensorflow_io as tfio
from tensorflow.python.framework.ops import disable_eager_execution
from siamese_network import build_network, projection_layer, prediction_layer
import csv
from debugging import plot_sample_preprocessing, plot_sample_input_training, networkout_to_numpy, plot_std_mean_network_output, plot_PCA, plot_TSNE, plot_classes, audio_to_spec_tensor


def calculate_accuracy(y_pred, y_true):
    sum_all = np.dot(y_pred, y_true)
    print('Accuracy is {}'.format(sum_all/len(y_pred)))


def calc_center_res(out1):
    m_o = []
    m_r = []
    center_1 = np.mean(out1, axis=0)
    for line in out1:
        residual_1 = line - center_1

        m_o.append(np.linalg.norm(center_1, ord=2) / np.linalg.norm(out1, ord=2))
        m_r.append(np.linalg.norm(residual_1, ord=2) / np.linalg.norm(out1, ord=2))

    print("M_o is {}".format(np.mean(np.asarray(m_o))))
    print("M_r is {}".format(np.mean(np.asarray(m_r))))

    return m_o, m_r


def neg_cosine_similarity(p, z):
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

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


def full_loss_v2(pred1, pred2, out1, out2):

    #return 0.5 * (tf.keras.losses.mean_squared_error(pred1, tf.stop_gradient(out2)) + tf.keras.losses.mean_squared_error(pred2, tf.stop_gradient(out1)))
    return neg_cosine_similarity(pred1, out2) / 2 + neg_cosine_similarity(pred2, out1) / 2


def get_o_and_r(out1, out2):

    center_1 = tf.reduce_mean(out1, axis=0)
    center_2 = tf.reduce_mean(out2, axis=0)

    residual_1 = tf.math.subtract(out1, center_1)
    residual_2 = tf.math.subtract(out2, center_2)

    m_o1 = tf.norm(center_1, ord=2) / tf.norm(out1, ord=2)
    m_r1 = tf.norm(residual_1, ord=2) / tf.norm(out1, ord=2)

    m_o2 = tf.norm(center_2, ord=2) / tf.norm(out2, ord=2)
    m_r2 = tf.norm(residual_2, ord=2) / tf.norm(out2, ord=2)

    m_o = (m_o1 + m_o2) / 2
    m_r = (m_r1 + m_r2) / 2

    return m_o, m_r


class CustomModel(tf.keras.Model):
    def compile(self, optimizer, loss, metric):
        super().compile(optimizer)
        self.loss = loss
        self.metric = metric

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x1, x2, y = data

        with tf.GradientTape() as tape:
            pred1, out1 = self(x1, training=True)
            pred2, out2 = self(x2, training=True)

            loss = self.loss(pred1, pred2, out1, out2)
            # class_loss = (tf.keras.losses.categorical_crossentropy(label, class_label1) + tf.keras.losses.categorical_crossentropy(label, class_label2))/2
            cosine_loss = full_loss(pred1, pred2, out1, out2)
            m_o, m_r = get_o_and_r(out1, out2)

            loss = loss + (1 - m_r) + m_o

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #std = 0.5 * tf.math.reduce_std(tf.math.l2_normalize(out1, axis=0), axis=1) + 0.5 * tf.math.reduce_std(
        #    tf.math.l2_normalize(out2, axis=0))

        # Update metrics (includes the metric that tracks the loss)
        loss_tracker.update_state(loss)
        #train_acc_metric.update_state(std)
        train_center_metric.update_state(m_o)
        train_r_metric.update_state(m_r)
        train_cosine_metric.update_state(cosine_loss)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result(), "center": train_center_metric.result(), "residual": train_r_metric.result(), "cosine_loss": train_cosine_metric.result()}


if __name__ == "__main__":

    tf.keras.backend.clear_session()

    log_dir = "logs/fit/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    metadata = os.path.join(log_dir, 'metadata.tsv')

    audio_path = 'X:/sounds/VoxCeleb/vox1_dev_wav/wav/id10743/N94AAE1xEuE/00001.wav'
    model_name = 'SiamesesMyLoss'

    DEBUG = 0

    if DEBUG:
        print('Debugging Mode is turned on')
        #plot_sample_preprocessing(audio_path)

    train_siamese = False

    # Settings for Extraction
    fs = 22050
    audio_length = 2.5
    window_length = int(fs*0.025)
    hop_length = int(fs*0.01)

    # Settings for Training
    EPOCHS = 15
    OUTPUT_DIMENSION = 512

    loss_tracker = tf.keras.metrics.Mean(name="loss")
    train_acc_metric = tf.keras.metrics.Mean()
    train_center_metric = tf.keras.metrics.Mean()
    train_r_metric = tf.keras.metrics.Mean()
    train_class_metric = tf.keras.metrics.Mean()
    train_cosine_metric = tf.keras.metrics.Mean()

    if "scratch" in os.getcwd():
        tfrecord_files = glob.glob('/scratch/work/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        BATCH_SIZE = 64
        SHUFFLE_BUFFER_SIZE = 100
        #mixed_precision.set_global_policy('mixed_float16')

        print('Starting training on cluster')
    else:
        #tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebTFRecordsAudio/*.tfrecords')
        tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebTFRecordAudioTest/*.tfrecords')
        #tfrecord_files = tfrecord_files[:5]
        BATCH_SIZE = 50
        SHUFFLE_BUFFER_SIZE = 100
        #mixed_precision.set_global_policy('mixed_float16')

        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if DEBUG:
        abc = 1

    #print("Test Vectors")
    #test_vec1 = tf.random.uniform((20, 512))
    #test_vec2 = tf.random.uniform((20, 512))
    #test_o_and_r = get_o_and_r(test_vec1, test_vec2)

        # , profile_batch=(50, 100)
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
        model_name,
        monitor="loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    if train_siamese:

        learning_rate = 0.05 * BATCH_SIZE / 256
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(learning_rate, 1000)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

        if DEBUG:
            tfrecord_files = glob.glob('C:/Users/rechs1/VoxCelebTFRecordAudioTest/*.tfrecords')
        # Parameters for training the siamese network

        # Get training data
        train_dataset = get_dataset(tfrecord_files, SHUFFLE_BUFFER_SIZE, BATCH_SIZE, window_length=window_length, hop_length=hop_length, phase='train', fs=fs)

        w, x, y, z = get_input_shape(train_dataset)
        input_a = tf.keras.Input(shape=(x, y, z))
        input_b = tf.keras.Input(shape=(x, y, z))

        if DEBUG == 1:
            plot_dataset = get_dataset(tfrecord_files[0], SHUFFLE_BUFFER_SIZE, BATCH_SIZE, window_length=window_length,
                                        hop_length=hop_length, phase='plot', fs=fs)
            plot_sample_input_training(plot_dataset)

        # Build network
        resnet_out = build_network(input_a, OUTPUT_DIMENSION)

        in1 = resnet_out(input_a)
        in2 = resnet_out(input_b)

        proj1 = projection_layer(in1, OUTPUT_DIMENSION)
        pred1 = prediction_layer(proj1, OUTPUT_DIMENSION)
        #class1 = tf.keras.layers.Dense(512, activation="relu")(pred1)
        #class_label = tf.keras.layers.Dense(40, activation="softmax")(class1)

        print('-----------------------Training Starting---------------------')

        model = tf.keras.Model([input_a, input_b], [pred1, proj1])

        model.summary()
        model.compile(optimizer=optimizer, loss=full_loss_v2, metric=[train_acc_metric, train_center_metric, train_r_metric,
        train_class_metric,
        train_cosine_metric] )
        #model.summary()
        #if DEBUG:

        print('--------------------- Training Starting --------------------------')
        if "scratch" not in os.getcwd():
            model.fit(x=train_dataset, epochs=EPOCHS, verbose=1, callbacks=[tensorboard_callback, modelcheckpoint])
        else:
            model.fit(x=train_dataset, epochs=EPOCHS, verbose=1, callbacks=[modelcheckpoint])

        model.save(model_name)
        print('Model successfully saved under the name: {}'.format(model_name))
    #
    else:
        model = tf.keras.models.load_model(model_name, custom_objects={'full_loss': full_loss})
        #model.summary()

    # Load Test Set
    print('Starting verification validation')
    #tfrecord_files_test = glob.glob('C:/Users/rechs1/VoxCelebTFRecordAudioTest/*.tfrecords')
    #test_dataset, validation_dataset = get_test_dataset(tfrecord_files_test, SHUFFLE_BUFFER_SIZE=100, BATCHSIZE=128)

    #if DEBUG:#

        #siamese_predictions, labels = networkout_to_numpy(test_dataset, model, OUTPUT_DIMENSION=2048)
        #plot_std_mean_network_output(siamese_predictions, labels)
        #plot_PCA(siamese_predictions, labels)
        #plot_TSNE(siamese_predictions, labels)

    # Build and train classification layer
    model.trainable = False
    model.summary()

    # Read in validation pairs
    with open('VoxCeleb-1_validation_trials.txt') as f:
        name1, name2, label = tools.read_trials(f)

    if "scratch" in os.getcwd():
        validation_folder = "/scratch/work/rechs1/VoxCelebValidation/wav/"
    else:
        validation_folder = "C:/Users/rechs1/VoxCelebValidation/wav/"

    length_train = len(name1)

    label_arr = np.asarray(label[:length_train], dtype=int)
    label_arr_ext = np.reshape(label_arr, (-1, 1))

    name1_dataset = tf.data.Dataset.from_tensor_slices(name1[:length_train])
    name2_dataset = tf.data.Dataset.from_tensor_slices(name2[:length_train])
    label_dataset = tf.data.Dataset.from_tensor_slices(label_arr_ext)
    name1 = name1[:length_train]
    name2 = name2[:length_train]

    name1_dataset = name1_dataset.map(lambda x: audio_to_spec_tensor(x, window_length=window_length, hop_length=hop_length, fs=fs)).batch(100).prefetch(tf.data.AUTOTUNE)
    name2_dataset = name2_dataset.map(lambda x: audio_to_spec_tensor(x, window_length=window_length, hop_length=hop_length, fs=fs)).batch(100).prefetch(tf.data.AUTOTUNE)

    ds = tf.data.Dataset.zip((name1_dataset, name2_dataset))
    ds_train = tf.data.Dataset.zip((ds, label_dataset))
    # Predict and calculate siamese similarity between pairs
    w, x, y, z = get_input_shape(name1_dataset, phase='eval')

    model_in1 = tf.keras.Input(shape=(x, y, z))
    model_in2 = tf.keras.Input(shape=(x, y, z))

    output_encoder = model.get_layer("backbone").output
    representation_model = tf.keras.Model(model.input, output_encoder)

    #weights = representation_model.get_layer('OutputPrediction').get_weights()[0]
    # name of the tensor.
    #checkpoint = tf.train.Checkpoint(representation_model=weights)
    #checkpoint.save(os.path.join(log_dir, "representation.ckpt"))

    # Set up config.
    #config = projector.ProjectorConfig()
    #embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    #embedding.tensor_name = "representation/.ATTRIBUTES/VARIABLE_VALUE"
    #mbedding.metadata_path = 'metadata.tsv'
    #rojector.visualize_embeddings(log_dir, config)

    represent1 = representation_model(model_in1)
    represent2 = representation_model(model_in2)

    last_output1 = tf.keras.layers.Dense(512, activation="relu")(represent1)
    last_output2 = tf.keras.layers.Dense(512, activation="relu")(represent2)

    last_output = tf.keras.layers.Subtract()([last_output1, last_output2])
    last_output = tf.keras.layers.Dense(1, activation='sigmoid')(last_output)

    verification_model = tf.keras.Model([model_in1, model_in2], last_output)

    verification_model.summary()

    verification_model.compile(optimizer="adam", loss=tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
    verification_model.fit(ds_train, epochs=2, callbacks=tensorboard_callback)

    out1 = representation_model.predict(name1_dataset, verbose=1)
    out2 = representation_model.predict(name2_dataset, verbose=1)

    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        name1_visualize = name2[:1500]
        for label in name1_visualize:
            f.write("{}\n".format(get_id(label)))

    out2_visualize = out2[:1500]
    np.savetxt(os.path.join(log_dir, 'vectors.tsv'), out2_visualize, delimiter="\t")

    siamese_score = verification_model.predict(ds_train)

    #representation_arr = {}
    #for i in range(len(name1)):
    #   id_1 = get_id(name1[i])
    #    if id_1 not in representation_arr.keys():
    #       representation_arr[id_1] = [out1[i]]
    #    else:
    #        representation_arr[id_1].append(out1[i])#

    #for i in range(len(name2)):
    #    id_2 = get_id(name2[i])
    #    if id_2 not in representation_arr.keys():
    #        representation_arr[id_2] = [out2[i]]
    #    else:
    #        representation_arr[id_2].append(out2[i])

    #test = calc_center_res(out1)

    print("Calculating Scores")

    scores = []
    eucl_distance = []
    for k in range(len(out1)):
        check1 = out1[k]
        chec2 = out2[k]
        cos_sim = np.dot(out1[k], out2[k]) / (np.linalg.norm(out1[k]) * np.linalg.norm(out2[k]))
        scores.append(cos_sim)
        eucl_distance.append(np.linalg.norm(out1[k] - out2[k]))

    if DEBUG:
        # Plot the similarity and distance scores between similar and dissimilar pairs
        plt.boxplot(eucl_distance[::2], eucl_distance[1::2])
        plt.show()

    siamese_score = np.round(siamese_score)
    #scores = np.round(scores)
        # Plot the representation of the same speaker to see the intra difference between speakers

    out = open("scores.txt", 'w')
    for a in scores:
        out.write('%s\n' % a)
    out.close()
    print('..... Scores are written in: %s' % "scores.txt")

    tools.calculate_EER(scores)
    tools.calculate_EER(siamese_score)
    if DEBUG:
        print('DEBUG Ending')

    x = 1
