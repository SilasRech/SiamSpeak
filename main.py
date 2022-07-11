import glob
import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import librosa


def plot_tensor(tensor, mode):
    """
    :param tensor: audio tensorflow tensor that will be plotted
    :param mode: the dataformat of the audio tensor (either time for time domain, or freq for spectrogram)
    :return:
    """
    plt.figure()
    if mode == "time":
        plt.plot(tensor.numpy())
    elif mode == "freq":
        plt.imshow(tensor.numpy())
    else:
        raise ValueError("Mode not recognized, please choose between time or freq")
    plt.show()


def load_audio_as_augmented_mel(path):

    audio_tensor = tfio.audio.AudioIOTensor(path, dtype=tf.float32)
    audio_rate = tf.cast(audio_tensor.rate, tf.float32)
    length_audio = audio_rate * tf.constant(2.5, dtype=tf.float32)
    # Normalize Tensor
    audio_tensor = audio_tensor[:length_audio]
    audio_tensor = tf.squeeze(audio_tensor, axis=[-1])
    audio_norm = tf.cast(audio_tensor, tf.float32) / tf.cast(tf.math.reduce_max(audio_tensor), tf.float32)
    audio_nomean = audio_norm - tf.math.reduce_mean(audio_norm)

    # Transform to Mel Scale
    spec = tfio.audio.spectrogram(
        audio_nomean, nfft=1024, window=512, stride=512)

    mel_spec = tfio.audio.melscale(
        spec, rate=audio_rate, mels=128, fmin=0, fmax=8000)

    db_mel = tfio.audio.dbscale(mel_spec, top_db=80)

    # Spec Augment for Extractions
    freq_mask = tfio.audio.freq_mask(db_mel, param=10)

    time_mask = tfio.audio.time_mask(db_mel, param=10)

    #if DEBUG == 1:
    #    plot_tensor(audio_nomean, "time")
    #    plot_tensor(tf.math.log(spec), "freq")
    #    plot_tensor(db_mel, "freq")
    #    plot_tensor(freq_mask, "freq")
    #    plot_tensor(time_mask, "freq")

    return freq_mask, time_mask


def get_id(path):
    """
    :param path: Path to audiofile in the VoxCeleb Database
    :return: ID of the speaker of the audiofile (the name of the folder that contains the ID)
    """
    split_tensor = tf.strings.split(path, sep="/")

    return split_tensor[-3]


if __name__ == "__main__":

    DEBUG = 0
    files_per_TFRecord = 5

    #test_database = ["/work/t405/T40571/sounds/VoxCeleb/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav"]
    test_database = glob.glob("/work/t405/T40571/sounds/VoxCeleb/vox1_dev_wav/wav/**/*.wav", recursive=True)
    test_database = test_database[:10]

    dataset_data = tf.data.Dataset.from_tensor_slices(test_database)
    dataset_data = dataset_data.map(load_audio_as_augmented_mel)

    dataset_label = tf.data.Dataset.from_tensor_slices(test_database)
    dataset_label = dataset_label.map(get_id)

    dataset3 = tf.data.Dataset.zip((dataset_data, dataset_label))

    processed_features = dataset3.get_single_element()

    #  Test if path is working
    if len(test_database) == 0:
        raise ValueError('The path does not contain any audio files, please check the path')
    else:
        print("Found {} files in the path".format(len(test_database)))

    # Settings for Extraction
    window_length = 512

    # Testing:
    with tf.io.TFRecordWriter("/u/14/rechs1/unix/VoxCelebTFRecords/Test.tfrecords") as file_writer:
        for i in range(files_per_TFRecord):
            mel_element1, mel_element2 = load_audio_as_augmented_mel(test_database[i])
            x_shape = mel_element1.shape

            y = get_id(test_database[i])

            ds = (tf.data.Dataset.from_tensors(mel_element1)
                  .map(tf.io.serialize_tensor))

            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "x_1": tf.train.Feature(float_list=tf.train.FloatList(value=[])),
                "x_2": tf.train.Feature(float_list=tf.train.FloatList(value=[tf.reshape(mel_element2, [-1]).numpy().tolist()])),
                "y": tf.train.Feature(float_list=tf.train.BytesList(value=[y])),
                "shape_x": tf.train.Feature(float_list=tf.train.FloatList(value=[x_shape]))
            })).SerializeToString()
            file_writer.write(record_bytes)
            x = 1

    x = 1
