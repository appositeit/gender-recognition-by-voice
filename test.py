import pyaudio
import os
import wave
import librosa
import numpy as np
import tqdm

from sys import byteorder
from array import array
from struct import pack
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore', message='Unable to register cuBLAS factory:')
warnings.filterwarnings('ignore', message='E tensorflow/stream_executor/cuda/cuda_blas.cc:2981]')

import training_data
import vggish_input
import data_schema

THRESHOLD = 500
CHUNK_SIZE = 1024
#  FORMAT = pyaudio.paInt16
FORMAT = 4  # pyaudio.paInt24
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


def test_fileset(model_file, pset):
    '''This tests a training data set using the model and stores the test results in the
    "test" table, both original and new outcome. This will be used by a UI to allow the user
    to explore why particular audio samples are failing classification.'''

    tester = ModelTester(model_file=model_file)
    batch_size = 1000
    pdd, pdm = training_data.make_p_parameters('raw22', pset)

    # Load valid samples from the data schema
    gdb = data_schema.GenderDB(pdm)
    samples = gdb.read_valid_samples()

    test_run = {
        'model_filename': model_file,
    }
    test_run_id = gdb.add_test_run(test_run)

    features = []
    tests = []

    for i, sample in tqdm.tqdm(enumerate(samples), 'Testing samples', total=len(samples)):
    # for i, sample in enumerate(samples):
        audio, sr = librosa.load(sample.filename)
        features = vggish_input.waveform_to_examples(audio, sr)
        features = np.array(features).reshape((1,1,96,64))
        test_male = tester.test(features)
        test = {
            'test_run_id': test_run_id,
            'sample_id': sample.id,
            'sample_male': sample.male,
            'test_male': test_male
        }
        tests.append(test)

        if i % batch_size == 0: 
            gdb.add_tests(tests)
            tests = []

    gdb.add_tests(tests)


def interactive():
    import argparse
    parser = argparse.ArgumentParser(description='''Gender recognition script, this will load the model you trained,
                                    and perform inference on a sample you provide (either using your voice or a file)''')
    parser.add_argument("-m", "--model", help="The path to the saved model file")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()

    file = args.file
    if not file or not os.path.isfile(file):
        # if file not provided, or it doesn't exist, use your voice
        print("Please talk")
        # put the file name here
        file = "test.wav"
        # record the file (start talking)
        record_to_file(file)
    # extract features and reshape it
    # features = extract_feature(file, mel=True).reshape(1, -1)
    audio, sr = librosa.load(file)
    features = vggish_input.waveform_to_examples(audio, sr)
    features = np.array(features).reshape((1,1,96,64))

    tester = ModelTester('vgg2')
    male_prob = tester.test(features)
    # show the result!
    # print("Result:", gender)
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    print(f"Male: {male_prob}\tFemale: {female_prob}")


class ModelTester:
    
    def __init__(self, model_label=None, model_file=None):
        # load the saved model (after training)
        # model = pickle.load(open("result/mlp_classifier.model", "rb"))
        if model_label:
            model_files, model_creator = training_data.list_models_and_creator(model_label)
            model_file = model_files[0]

            if model_creator:
                # construct the model
                self.model = model_creator()
                # load the saved/trained weights
                self.model.load_weights(model_files[1])

        self.model = load_model(model_file)

    def test(self, features):
        # predict the gender!
        # male_prob = model.predict(features)[0][0]
        return self.model(features, training=False)[0][0]

def main():
    model_file = '/mnt/fastest/jem/ml/training_data/models/by_run/wise-rain-110.h5'
    test_fileset(model_file, 'test')


if __name__ == "__main__":
    main()