import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_sample_preprocessing(audio_path):

        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        plt.plot(audio)
        plt.show()

        spectrogram = tfio.audio.spectrogram(tf.squeeze(audio), nfft=2040, window=256, stride=256)

        plt.imshow(spectrogram)
        plt.show()

        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=22050, mels=128, fmin=0, fmax=8000)

        plt.imshow(mel_spectrogram.numpy())
        plt.show()

        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

        plt.imshow(dbscale_mel_spectrogram.numpy())
        plt.show()

        spec1 = dbscale_mel_spectrogram[:len(dbscale_mel_spectrogram) // 2, :]
        plt.show()

        spec2 = dbscale_mel_spectrogram[len(dbscale_mel_spectrogram) // 2:, :]

        raw_audio1 = tf.expand_dims(tfio.audio.freq_mask(spec1, param=10), axis=2)
        plt.imshow(raw_audio1.numpy())
        plt.show()

        raw_audio2 = tf.expand_dims(tfio.audio.freq_mask(spec2, param=10), axis=2)
        plt.imshow(raw_audio2.numpy())
        plt.show()


def plot_sample_input_training(train_dataset):
        train_dataset.shuffle(100)
        test = train_dataset.take(3)
        f, ax = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all')
        counter_x = 0

        for x1, x2, y in test.as_numpy_iterator():
                ax[counter_x, 0].imshow(np.squeeze(x1[0]))
                ax[counter_x, 1].imshow(np.squeeze(x2[0]))
                ax[counter_x, 0].set_ylabel('Time [s]')
                ax[counter_x, 0].set_xlabel('Mel Frequency [Hz]')
                ax[counter_x, 1].set_ylabel('Time [s]')
                ax[counter_x, 1].set_xlabel('Mel Frequency [Hz]')

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