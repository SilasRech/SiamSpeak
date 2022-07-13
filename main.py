import glob
import os, datetime
import matplotlib.pyplot as plt
import librosa as rosa
import numpy as np
from tools import spec_augment
import tensorflow as tf
import siamese_network as sn
import tensorboard
from sklearn.preprocessing import OneHotEncoder


def plot_tensor(spec, mode):
    """
    :param tensor: audio tensorflow tensor that will be plotted
    :param mode: the dataformat of the audio tensor (either time for time domain, or freq for spectrogram)
    :return:
    """
    plt.figure()
    if mode == "time":
        plt.plot(spec)
    elif mode == "freq":
        plt.imshow(spec)
    else:
        raise ValueError("Mode not recognized, please choose between time or freq")
    plt.show()


def load_audio_as_augmented_mel(path, length):

    audio_file, fs = rosa.load(path)

    length_audio = int(fs * length)
    # Normalize Tensor
    audio_file = np.squeeze(audio_file)
    audio_file = audio_file[:length_audio]
    audio_norm = audio_file - np.mean(audio_file)
    audio_nomean = audio_norm / np.max(abs(audio_norm))

    # Transform to Mel Scale
    mel_spec = rosa.feature.melspectrogram(y=audio_nomean, sr=fs, n_fft=1024, hop_length=512, win_length=512)
    spec = rosa.power_to_db(mel_spec, ref=np.max)
    # Spec Augment for Extractions
    augmented1 = spec_augment(spec, 10)
    augmented2 = spec_augment(spec, 10)

    #if DEBUG == 1:
    #    plot_tensor(audio_nomean, "time")
    #    plot_tensor(tf.math.log(spec), "freq")
    #    plot_tensor(db_mel, "freq")
    #    plot_tensor(freq_mask, "freq")
    #    plot_tensor(time_mask, "freq")

    return augmented1, augmented2


def get_id(path):
    """
    :param path: Path to audiofile in the VoxCeleb Database
    :return: ID of the speaker of the audiofile (the name of the folder that contains the ID)
    """
    normalized_path = os.path.normpath(path)
    speaker_id = normalized_path.split(os.sep)[-3]

    return speaker_id[-4:]


def extract_features(files, audio_length):

    augmented_1_list = []
    augmented_2_list = []
    id_list = []

    for i in range(len(files)):
        augmented_spec1, augmented_spec2 = load_audio_as_augmented_mel(files[i], audio_length)
        speaker_id = get_id(files[i])

        augmented_1_list.append(np.expand_dims(augmented_spec1, axis=2))
        augmented_2_list.append(np.expand_dims(augmented_spec2, axis=2))
        id_list.append(speaker_id)
        print('Iteration {} of {}'.format(i, len(files)))

    return np.asarray(augmented_1_list), np.asarray(augmented_2_list), np.asarray(id_list)


def list_audio_files(preloaded, path):
    if preloaded:
        with open(path) as f:
           files = f.read().splitlines()
    else:
        files = glob.glob("X:/sounds/VoxCeleb/vox1_dev_wav/wav/**/*.wav", recursive=True)
        with open('path', 'w') as fp:
            for item in files:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')

    return files


def neg_cosine_similarity(p, z):

    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

    product = tf.math.reduce_sum(tf.math.multiply(p, z), axis=1)

    return -tf.math.reduce_mean(product)


def loss(model, x, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    output1, output2, pred1, pred2 = model(x, training=training)

    return neg_cosine_similarity(pred1, output2) / 2 + neg_cosine_similarity(pred2, output1) / 2


def grad(model, inputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def linear_prediction(input_shape):
    # MLP Projection Layer
    x = tf.keras.layers.Dense(2048, activation='relu')(input_shape)
    x = tf.keras.layers.BatchNormalization()(x)
    pred_class = tf.keras.layers.Dense(7, activation='softmax')(x)

    return tf.keras.Model(inputs=input_shape, outputs=pred_class)


def saveToTFRecords(files, name, tffilesize=100, audio_length=2.5, ):

    for k in range(len(files)//tffilesize):
        one_TFfile = files[k*tffilesize:(k+1)*tffilesize]
        train_features_1, train_features_2, y_train = extract_features(one_TFfile, audio_length)
        



if __name__ == "__main__":

    DEBUG = 0
    preloaded_files = True
    load_training_data = False

    # Settings for Extraction
    audio_length = 2.5
    window_length = 512
    hop_length = 512

    input_shape = (128, 108, 1)

    #test_database = ["X:/sounds/VoxCeleb/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav"]
    if load_training_data:
        file_list = list_audio_files(preloaded_files, 'speaker_files.txt')

        #  Test if path is working
        if len(file_list) == 0:
            raise ValueError('The path does not contain any audio files, please check the path')
        else:
            print("Found {} files in the path".format(len(file_list)))

        # Split into 80% Training Files

        #train_files = file_list[:1000]
        #test_files = file_list[1020:1100]

        split_index = int(0.8 * len(file_list))

        train_files = file_list[:split_index]
        test_files = file_list[split_index:]

        print('Extracting Training Features')
        train_features_1, train_features_2, y_train = extract_features(train_files, audio_length)
        print('Finished training features, extracing testing features')

        test_features_1, test_features_2, y_test = extract_features(test_files, audio_length)

        np.savez('VoxCeleb_Feature_for_Siamese_TrainingFullDataSet.npz', x_train1=train_features_1, x_train2=train_features_2, x_test1=test_features_1, x_test2=test_features_2, y_train=y_train, y_test=y_test)
    else:
        with np.load('VoxCeleb_Feature_for_Siamese_TrainingFullDataSet.npz') as data:
            train_examples1 = data['x_train1']
            train_examples2 = data['x_train2']

            test_examples1 = data['x_test1']
            test_examples2 = data['x_test2']

            train_labels = data['y_train']
            test_labels = data['y_test']

        enc = OneHotEncoder()
        hot_train_labels = enc.fit_transform(np.reshape(train_labels, (-1, 1))).toarray()

        train_dataset_x = tf.data.Dataset.from_tensor_slices((train_examples1, train_examples1))
        train_dataset_y = tf.data.Dataset.from_tensor_slices(hot_train_labels)
        dataset_train = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))

        BATCH_SIZE = 1
        SHUFFLE_BUFFER_SIZE = 1

        train_dataset = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1, tf.data.AUTOTUNE)

        siamese_net = sn.build_network(input_shape=input_shape)
        siamese_net.summary()

        simple_predictor = linear_prediction(tf.keras.Input((512)))
        simple_predictor.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"])

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

        train_loss_results = []
        train_accuracy_results = []

        num_epochs = 10

        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # TC = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        # TC.set_model(model)
        summary_writer = tf.summary.create_file_writer(logdir=log_dir)
        print('-----------------------Training Starting---------------------')
        for epoch in range(num_epochs):

            print('Starting Epoch: {} of {}'.format(epoch, num_epochs))
            epoch_loss_avg = tf.keras.metrics.Mean()

            # Training loop - using batches of 32
            for x, y in train_dataset:
                # Optimize the model
                loss_value, grads = grad(siamese_net, x)
                #print(loss_value.numpy())
                optimizer.apply_gradients(zip(grads, siamese_net.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            with summary_writer.as_default():
                tf.summary.scalar('epoch_loss_avg', epoch_loss_avg.result(), step=optimizer.iterations)
            # End epoch
            print('Average Loss in Epoch {}'.format(epoch_loss_avg.result().numpy()))
            train_loss_results.append(epoch_loss_avg.result())
            test = 1
        for x, y in train_dataset:
            predictions = siamese_net(x)

            simple_predictor.fit(predictions[0], y, batch_size=20, epochs=10)

        #test_dataset_x = tf.data.Dataset.from_tensor_slices((test_examples1, test_examples2))
        #test_dataset_y = tf.data.Dataset.from_tensor_slices(test_labels)
        #ataset_test = tf.data.Dataset.zip((test_dataset_x, test_dataset_y))

        #test_dataset = dataset_test.batch(BATCH_SIZE)

        # Evaluation
        model_new = tf.keras.Model(inputs=siamese_net.get_input_at(0), outputs=siamese_net.get_layer('OutputPrediction1').output)
        predictions = siamese_net(train_examples1)

        train_dataset_prediction = tf.data.Dataset.from_tensor_slices(predictions, hot_train_labels)




    # Testing

    #tensorboard --log dirlogs
    x = 1
