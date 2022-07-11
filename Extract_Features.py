from scipy import signal
import scipy.io.wavfile as wav
import pickle
import functions
import numpy as np
import librosa

train_data= np.full((129,597),0)

def spectrogram(filename):
    sample_rate, samples = wav.read(filename)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram

if __name__ == '__main__':
    file_lists = ('/l/Abraham/Projects/SpeakerVerification/Data/vox1_test_wav/wav/training_labels.lst')
    data_list = open(file_lists, "r")
    #pickle_file='train.pickle'
    data_names, data_labels = functions.read_file(data_list)
    for i in range(0, len(data_names)):
        pickle_file_name= data_names[i] + '.pickle'
        data=spectrogram(data_names[i] + '.wav')
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(data, pickle_file)
        pickle_file.close()

