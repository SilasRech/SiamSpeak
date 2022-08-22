import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def join_paths(path1, path2):
        return tf.strings.join([path1, path2], separator='/')


def audio_to_spec_tensor(path, window_length=512, hop_length=256, fs=22050):

        if "scratch" in os.getcwd():
                mainpath = "/scratch/work/rechs1/VoxCelebValidation/wav/"
        else:
                mainpath = "C:/Users/rechs1/VoxCelebValidation/wav"

        audio1 = join_paths(mainpath, path)
        audio1 = tf.strings.join([audio1, ".wav"], separator='')

        spec1 = extract_features(audio1, window_length=window_length, hop_length=hop_length, fs=fs)
        return spec1


def extract_features(path, window_length=512, hop_length=256, fs=22050):

        audio = tf.io.read_file(path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        length_audio = tf.cast(tf.multiply(tf.cast(fs, dtype=tf.float32), tf.constant(2.5, dtype=tf.float32)),
                               dtype=tf.int32)
        # Normalize Tensor
        audio = tf.squeeze(audio)
        audio = audio[:length_audio]

        spectrogram = tfio.audio.spectrogram(tf.squeeze(audio), nfft=1024, window=window_length, stride=hop_length)

        mel_spectrogram = tfio.audio.melscale(
                spectrogram, rate=fs, mels=128, fmin=50, fmax=8000)

        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

        dbscale_mel_spectrogram = dbscale_mel_spectrogram[:len(dbscale_mel_spectrogram) // 2, :]

        raw_audio1 = tf.expand_dims(dbscale_mel_spectrogram, axis=2)
        return raw_audio1


def plot_sample_preprocessing(audio_path):

        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        audio = audio[:tf.cast(16000*2.5, dtype=tf.int32)]
        plt.plot(audio)
        plt.show()

        spectrogram = tfio.audio.spectrogram(tf.squeeze(audio), nfft=512, window=512, stride=196)

        plt.imshow(spectrogram)
        plt.show()

        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

        plt.imshow(mel_spectrogram.numpy())
        plt.show()

        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

        #plt.imshow(dbscale_mel_spectrogram.numpy())
        #plt.show()

        #spec1 = dbscale_mel_spectrogram[:len(dbscale_mel_spectrogram) // 2, :]
        #plt.show()

        #spec2 = dbscale_mel_spectrogram[len(dbscale_mel_spectrogram) // 2:, :]

        raw_audio1 = tf.expand_dims(tfio.audio.freq_mask(dbscale_mel_spectrogram, param=10), axis=2)
        plt.imshow(raw_audio1.numpy())
        plt.show()

        raw_audio2 = tf.expand_dims(tfio.audio.time_mask(dbscale_mel_spectrogram, param=10), axis=2)
        plt.imshow(raw_audio2.numpy())
        plt.show()


def plot_sample_input_training(train_dataset):
        train_dataset.shuffle(100)
        test = train_dataset.take(3)
        f, ax = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all')
        counter_x = 0
        #for spec1, spec2, y in test.as_numpy_iterator():
        for spectrogram, dbscale_mel_spectrogram, spec1, spec2 in test.as_numpy_iterator():
                #spec1 = np.flip(np.squeeze(spec1[0]).T, axis=0)
                #pec2 = np.flip(np.squeeze(spec2[0]).T, axis=0)

                ax[counter_x, 0].imshow(np.squeeze(spec1[0]))
                ax[counter_x, 1].imshow(np.squeeze(spec2[0]))
                ax[counter_x, 0].set_ylabel('Frames')
                ax[counter_x, 0].set_xlabel('Mel Frequency')
                ax[counter_x, 1].set_ylabel('Frames')
                ax[counter_x, 1].set_xlabel('Mel Frequency')
                counter_x += 1

        f.show()


def networkout_to_numpy(test_dataset, model, OUTPUT_DIMENSION=2048):
        examples = 15
        test = test_dataset.take(examples)
        siamese_predictions = np.zeros((0, OUTPUT_DIMENSION))
        labels_for_color = []

        for x1, y in test.as_numpy_iterator():
                output_prediction = model.predict(x1)
                siamese_predictions = np.vstack((siamese_predictions, output_prediction[0]))

                asdasdasdasd = np.argmax(y, axis=1)
                labels_for_color.append(np.asarray(asdasdasdasd))

        labels = np.reshape(np.asarray(labels_for_color), (examples * 50))
        return siamese_predictions, labels


def plot_std_mean_network_output(predictions, labels):
        variance_predictions = np.var(predictions, axis=0)
        mean_predictions = np.mean(predictions, axis=0)
        max_predictions = np.max(predictions, axis=0)
        min_predicitions = np.min(predictions, axis=0)

        mean_one_class = []
        variance_one_class = []

        for k in range(0, 40):
            find_a_class = np.where(labels == k)
            find_class_values = predictions[find_a_class]
            variance_one_class.append(np.var(find_class_values))
            mean_one_class.append(np.mean(find_class_values))
            max_predictions_one_class = np.max(find_class_values)
            min_predicitions_one_class = np.min(find_class_values)

        x_pos = np.arange(40)

        fig, ax = plt.subplots()
        ax.bar(x_pos, mean_one_class, yerr=variance_one_class, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_pos)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.show()


def plot_PCA(predictions, labels):
        pca = PCA(n_components=2).fit_transform(predictions)
        X_embedded = pca
        x_embed_X = X_embedded[:, 0]
        x_embed_Y = X_embedded[:, 1]
        plt.scatter(x=x_embed_X, y=x_embed_Y, c=labels)
        plt.show()


def plot_TSNE(predictions, labels):
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(predictions)
        x_embed_X = X_embedded[:, 0]
        x_embed_Y = X_embedded[:, 1]

        plt.scatter(x=x_embed_X, y=x_embed_Y, c=labels)
        plt.show()


def plot_classes(prediction, label):
        plt.hist(label)
        plt.show()

        plt.hist(prediction)
        plt.show()