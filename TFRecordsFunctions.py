import tensorflow as tf
import librosa as rosa

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
        'label': _int64_feature(label)
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


def write_images_to_tfr_short(images, labels, filename:str="audio"):
    filename = filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) # create a writer that'll store our data to disk
    count = 0

    for index in range(len(images)):
        # Get the data we want to write
        current_image = images[index]
        current_label = labels[index]

        out = parse_single_image(audio1=audio1, audio2= label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count