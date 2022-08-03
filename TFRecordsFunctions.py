import tensorflow as tf
import librosa as rosa
import numpy as np
import glob
import os
from tools import spec_augment, power_to_db
import glob
import tensorflow_io as tfio
import random


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_single_image(audio1, audio2, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        'height': _int64_feature(audio1.shape[0]),
        'width': _int64_feature(audio2.shape[1]),
        'raw_audio1': _bytes_feature(serialize_array(audio1)),
        'raw_audio2': _bytes_feature(serialize_array(audio2)),
        'label': _int64_feature(int(label))
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def parse_single_audio(audio1, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        'height': _int64_feature(audio1.shape[0]),
        'raw_audio1': _bytes_feature(serialize_array(audio1)),
        'label': _int64_feature(int(label))
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def get_id(path):
    """
    :param path: Path to audiofile in the VoxCeleb Database
    :return: ID of the speaker of the audiofile (the name of the folder that contains the ID)
    """
    normalized_path = os.path.normpath(path)
    speaker_id = normalized_path.split(os.sep)[-3]

    return speaker_id[-4:]


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


def write_images_to_tfr_short(audio_files, filename:str="audio"):
    filename = filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) # create a writer that'll store our data to disk
    count = 0

    audio_length = 2.5

    for i in range(len(audio_files)):
        # Get the data we want to write
        augmented_spec1, augmented_spec2 = load_audio_as_augmented_mel(audio_files[i], audio_length)
        speaker_id = int(get_id(audio_files[i]))-270

        out = parse_single_image(audio1=augmented_spec1, audio2=augmented_spec2, label=speaker_id)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def write_images_to_tfr_audio(audio_files, filename:str="audio"):
    filename = filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) # create a writer that'll store our data to disk
    count = 0

    audio_length = 2.5

    for audio_file in audio_files:
        print(count)
        # Get the data we want to write
        speaker_id = int(get_id(audio_file)) - 270
        audio_file1, fs = rosa.load(audio_file)

        length_audio = int(fs * audio_length)
        # Normalize Tensor
        audio_file1 = np.squeeze(audio_file1)
        audio_file1 = audio_file1[:length_audio]
        audio_norm = audio_file1 - np.mean(audio_file1)
        audio_nomean = audio_norm / np.max(abs(audio_norm))

        out = parse_single_audio(audio1=audio_nomean, label=speaker_id)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def list_audio_files(preloaded, path):
    if preloaded:
        with open(path) as f:
           files = f.read().splitlines()
    else:
        files = glob.glob(path, recursive=True)
        with open('path', 'w') as fp:
            for item in files:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')

    return files


def parse_tfr_element(element, phase, model=None):
    # use the same structure as above; it's kinda an outline of the structure we now want to create

    spectr = False
    # get our 'feature'-- our image -- and reshape it appropriately
    if phase == "train":
        if spectr:

            data = {
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
                'raw_audio1': tf.io.FixedLenFeature([], tf.string),
                'raw_audio2': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            content = tf.io.parse_single_example(element, data)

            height = content['height']
            width = content['width']
            label = content['label']
            raw_audio1 = content['raw_audio1']
            raw_audio2 = content['raw_audio2']

            raw_audio1 = tf.io.parse_tensor(raw_audio1, out_type=tf.double)
            raw_audio2 = tf.io.parse_tensor(raw_audio2, out_type=tf.double)

            raw_audio1 = tf.reshape(raw_audio1, shape=[height, width])
            raw_audio2 = tf.reshape(raw_audio2, shape=[height, width])

            return raw_audio1, raw_audio2, label
        else:
            data = {
                'height': tf.io.FixedLenFeature([], tf.int64),
                'raw_audio1': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            content = tf.io.parse_single_example(element, data)

            label = content['label']
            audio = content['raw_audio1']

            audio = tf.io.parse_tensor(audio, out_type=tf.float32)

            spectrogram = tfio.audio.spectrogram(tf.squeeze(audio), nfft=2040, window=256, stride=256)

            mel_spectrogram = tfio.audio.melscale(
                spectrogram, rate=22050, mels=128, fmin=0, fmax=8000)

            dbscale_mel_spectrogram = tfio.audio.dbscale(
                mel_spectrogram, top_db=80)

            spec1 = dbscale_mel_spectrogram[:len(dbscale_mel_spectrogram) // 2, :]
            spec2 = dbscale_mel_spectrogram[len(dbscale_mel_spectrogram) // 2:, :]

            spec1 = spec1[:96, :]
            spec2 = spec2[:96, :]
            #
            raw_audio1 = tf.expand_dims(tfio.audio.freq_mask(spec1, param=10), axis=2)
            raw_audio2 = tf.expand_dims(tfio.audio.time_mask(spec2, param=10), axis=2)

            label = tf.one_hot(label, depth=40)

            return raw_audio1, raw_audio2, label
    else:
        data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'raw_audio1': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        }
        content = tf.io.parse_single_example(element, data)

        label = content['label']
        audio = content['raw_audio1']

        audio = tf.io.parse_tensor(audio, out_type=tf.float32)

        spectrogram = tfio.audio.spectrogram(tf.squeeze(audio), nfft=2040, window=256, stride=256)

        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=22050, mels=128, fmin=0, fmax=8000)

        dbscale_mel_spectrogram = tfio.audio.dbscale(
            mel_spectrogram, top_db=80)

        spec1 = dbscale_mel_spectrogram[:len(dbscale_mel_spectrogram) // 2, :]

        spec1 = spec1[:96, :]
        #
        raw_audio1 = tf.expand_dims(spec1, axis=2)

        label = tf.one_hot(label, depth=40)
        return raw_audio1, label


def get_dataset(files, SHUFFLE_BUFFER_SIZE=100, BATCH_SIZE=20):
    # create the dataset
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)

    # pass every single feature through our mapping function
    dataset = dataset.map(lambda x: parse_tfr_element(x, "train"), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def get_test_dataset(tfrecord_files_test, model, SHUFFLE_BUFFER_SIZE=100, BATCHSIZE=25):
    tfrecord_files_class_train = tfrecord_files_test[:3]
    tfrecord_files_class_eval = tfrecord_files_test[-1]

    test_dataset = tf.data.TFRecordDataset(tfrecord_files_class_train)
    validation_dataset = tf.data.TFRecordDataset(tfrecord_files_class_eval)

    test_dataset = test_dataset.map(lambda x: parse_tfr_element(x, phase='eval', model=model)).shuffle(
        SHUFFLE_BUFFER_SIZE).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(lambda x: parse_tfr_element(x, phase='eval', model=model)).shuffle(
        SHUFFLE_BUFFER_SIZE).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

    return test_dataset, validation_dataset


def get_id(path):
    """
    :param path: Path to audiofile in the VoxCeleb Database
    :return: ID of the speaker of the audiofile (the name of the folder that contains the ID)
    """
    normalized_path = os.path.normpath(path)
    speaker_id = normalized_path.split(os.sep)[-3]

    return speaker_id[-4:]


if __name__ == "__main__":

    write_spectrum = False

    #dev_train = "X:/sounds/VoxCeleb/vox1_dev_wav/wav/**/*.wav"
    dev_test = "X:/sounds/VoxCeleb/vox1_test_wav/wav/**/*.wav"
    speaker_list = 'speaker_files.txt'

    audio_length = 2.5
    files = list_audio_files(False, dev_test)
    random.shuffle(files)

    print('Found {} files'.format(len(files)))
    tffilesize = 1000

    # Continue from file 63 for spectrum
    # Continue from file 107 for audio based
    if write_spectrum:
        for k in range(63, len(files)//tffilesize):
            print('Writing TFRecord file {} of {}'.format(k+1, len(files)//tffilesize ))
            one_TFfile = files[k*tffilesize:(k+1)*tffilesize]
            write_images_to_tfr_short(one_TFfile, 'Z:/VoxCelebTFRecords/VoxCeleb_Set{}'.format(k))
    else:
        for k in range(0, len(files)//tffilesize):
            print("------------ Writing Audio Files to TFRecord --------------")
            print('Writing TFRecord file {} of {}'.format(k+1, len(files)//tffilesize ))
            one_TFfile = files[k*tffilesize:(k+1)*tffilesize]
            write_images_to_tfr_audio(one_TFfile, 'C:/Users/rechs1/VoxCelebTFRecordAudioTest/TFRecordAudioSet{}'.format(k))
        test = 1