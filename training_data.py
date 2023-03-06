'''Contains metadata about the training data itself. Used to manage
and generate training data.

Training data is processed in stages:
    * source: Source data.
    * processed: Source data is processed into some intermediary format.
        - Intermediary data comes in two major formats:
            * "Raw" is 1 second long 22Khz samples in wav format.
            * "xxx" a 128 byte long string that represents the something or
                other spectrum.
    * bundled: Intermediary data is bundled ready for training. This bundling
        process tries to account for the amount of RAM available in the GPU
        for training, so the training is done in multiple passes if necessary.

It makes sense to have separate directory structures for each of these stages
because original data should become comingled along the way e.g. common_voice
original data should eventuall end up bundled along with other data sources.

Processed data can reasonably follow a similar structure to the original source
data, but during bundling we want to mix training data from different sources.

There needs to be a chain of relationships which connect the eventual training
data with the original source data.

## Source

Source data should be sorted into train, test, and validate sets. We do NOT
want to muddy those data sets. So from the start we make distinct data
sets and never cross the streams.

As much as possible we want individual speakers in one of those pools, not
spread across the pools.

## Processed

Processed data sets simply take the original sample, conducts some intermediary
process, and produces some intermediate representation. See raw22 below for an
example. Processed data of each set type (train, test, validate) is stored
in a single directory. I think that's reasonable, naming and metadata should
allow us to map back to the Source.

Processing can allow us to reduce the dependency of the ML models on certain
features of voice (e.g. by frequency shifting samples) or make it more robust
(by introducing noise).


### raw22
The typical function on source data for preparing raw22 is taking staggered
1s samples (e.g. 0ms offset, 200ms offset, 250ms offset, 300ms offset),
checking to make sure some sound is present, and subsampling to 22kHz. This
file is then saved as a processed sample, a record is writen in the metadata
and continue


# Bundling

Bundling makes for fast loading during training, and allows us to mix up our
training data. The key inputs are:
    * How big should each bundle be?
    * What processed data sources are being used?
    * How are the processed data sourcees being combined?

I think it would be useful to give a target to the bundling algorithm, then
let it experiment to determine roughly how many samples will make a file of
that size. Note that we don't actually know how big the in memory
representation will be, but we can scale until we find something that works.

Data sources are defined in the SOURCE_DATA structure
'''

from collections import defaultdict
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] ="2"
import glob
import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import tqdm

from sklearn.model_selection import train_test_split

import model_functions
import cv_processor


LABEL2INT = {"male": 1, "female": 0}


'''Desscribes the original source training data.'''
ORIGINAL_DATA = {
    "cv": {
        "description": """Mozilla commonn voice data set.""",
        "base_path": "/mnt/working/jem/ml/source/common_voice/",
        "sets": {
            'train': {
                "path": 'cv-valid-train',
                "metadata": 'cv-valid-train.csv',
                },
            'test': {
                "path": 'cv-valid-test',
                "metadata": 'cv-valid-test.csv',
                },
            'validate': {
                "path": 'cv-valid-dev',
                "metadata": 'cv-valid-dev.csv',
                },
            }
    },
    "fl": {
        "description": """Multilingual voice sample set.""",
        "base_path": "/mnt/working/jem/ml/source/fl/",
        "sets": {
            'train': {
                "path": 'train',
                "metadata": '',
                },
            'test': {
                "path": 'test',
                "metadata": '',
                },
            }
    },
}

PROCESSED_DATA = {
    "raw22": {
        "description": """1 second long 22khz (22050 samples) wav files.""",
        "base_path": "/mnt/fastest/jem/ml/training_data/processed/raw22",
        "sets": {
            'train': {
                "path": 'train',
                "metadata": 'train.sqlite',
                },
            'test': {
                "path": 'test',
                "metadata": 'test.sqlite',
                },
            'validate': {
                "path": 'validate',
                "metadata": 'validate.sqlite',
                },
            }
        }
}

# PROCESSED2_DATA = {
#     "vgg22": {
#         "description": """VGGish params (an array of mel spectograms) based raw22 samples""",
#         "base_path": "/mnt/fastest/jem/ml/training_data/processed/vgg22",
#         "sets": {
#             'train': {
#                 "path": 'train',
#                 },
#             'test': {
#                 "path": 'test',
#                 },
#             'validate': {
#                 "path": 'validate',
#                 },
#             },
#         },
# }

BUNDLED_DATA = {
    "raw22": {
        "description": """1 second long 22khz (22050 samples) wav files,
        composited into NUMPY arrays.""",
        "base_path": "/mnt/fastest/jem/ml/training_data/bundled",
        "sets": {
            'train': {
                "path": 'train',
                "base_features_filename": 'train_features',
                "base_labels_filename": 'train_labels',
                },
            'test': {
                "path": 'test',
                "base_features_filename": 'test_features',
                "base_labels_filename": 'test_labels',
                },
            'validate': {
                "path": 'validate',
                "base_features_filename": 'validate_features',
                "base_labels_filename": 'validate_labels',
                },
        },
    },
    "vgg22": {
        "description": """VGGish params (an array of mel spectograms) based raw22 samples""",
        "base_path": "/mnt/fastest/jem/ml/training_data/bundled/vgg22",
        "sets": {
            'train': {
                "path": 'train',
                "base_features_filename": 'vgg_features',
                "base_labels_filename": 'vgg_labels',
                },
            'test': {
                "path": 'test',
                "base_features_filename": 'vgg_features',
                "base_labels_filename": 'vgg_labels',
                },
            'validate': {
                "path": 'validate',
                "base_features_filename": 'vgg_features',
                "base_labels_filename": 'vgg_labels',
                },
        },
    },
}


'''Describes the ML training models, including how to create them and
what sort of training data they consume.'''
MODELS = {
    "base_path": "/mnt/fastest/jem/ml/training_data/models",
    "raw": {
        "description": "Train the model on a raw 1s 22kHz sample.",
        "create_model": model_functions.create_raw_model,
        "model_data": "raw22",
        "filename" : "model_raw.h5"
    },
    "lstm": {
        "description": """Train the model on a raw 1s 22kHz sample with an LSTM
            model.""",
        "create_model": model_functions.create_lstm_model,
        "model_data": "raw22",
        "filename" : "model_lstm.h5"
    },
    "4layer": {
        "description": """Train the model on a raw 1s 22kHz sample with a 4layer
            ML model.""",
        "create_model": model_functions.create_4layer_model,
        "model_data": "raw22",
        "filename" : "model_4layer.h5"
    },
    "5layer": {
        "description": """Train the model on a raw 1s 22kHz sample with a
            5layer ML model.""",
        "create_model": model_functions.create_5layer_model,
        "model_data": "raw22",
        "filename" : "model_5layer.h5"
    },
    "6layer": {
        "description": """Train the model on a raw 1s 22kHz sample with a
            6layer ML model.""",
        "create_model": model_functions.create_6layer_model,
        "model_data": "raw22",
        "filename" : "model_6layer.h5"
    },
    "vgg1": {
        "description": """Create a VGGish model.""",
        "create_model": model_functions.create_vgg_cnn,
        "model_data": "vgg22",
        "filename" : "model_vgg1",
        "filename_suffix" : ".h5",
    },
    "vgg2": {
        "description": """Create a VGGish model with an extra intermediary layer.""",
        "create_model": model_functions.create_vgg_cnn,
        "model_data": "vgg22",
        "filename" : "model_vgg2",
        "filename_suffix" : ".h5",
    },
}


# Processors take the following inputs:
# * Source data dir
# * Source metadata file
# * Destination processed dir
# * Destination metadata file
# # Optional dict of processing behaviours
#
# The proessing behaviours include:
# - Force rewrite: Clear the Destination metadata and fully regenerate the
#   dataset.
ORIGINAL_TO_PROCESSED = {
    "cv": {
        "raw22": {
            "description": """Transform cv source data to raw22 processed
            data""",
            # "processor": cv_processor.generate_raw22,
        }
    }


}


def make_o_parameters(source, oset):
    """Generate the basic parameters for the op processor."""
    sp = ORIGINAL_DATA[source]

    base_path = sp["base_path"]
    p = sp["sets"][oset]["path"]
    sdd = os.path.join(base_path, p, p)
    sdm = os.path.join(base_path, sp["sets"][oset]["metadata"])
    return sdd, sdm


def make_p_parameters(format, pset):
    pp = PROCESSED_DATA[format]

    p = pp["sets"][pset]["path"]
    pdd = os.path.join(pp["base_path"], p)
    pdm = os.path.join(pp["base_path"], pp["sets"][pset]["metadata"])

    return pdd, pdm

def make_p2_parameters(format, pset):
    pp = PROCESSED2_DATA[format]

    p2dm = os.path.join(pp['base_path'], pp["sets"][pset]["path"])

    return p2dm


def make_b_parameters(format, bset):
    '''make_b_parameters

    blf: Base labels filenamery: base path for bundled data
    bff: Bundled Features Filename
    blf: Base Labels Filename
    '''
    bp = BUNDLED_DATA[format]

    p = bp["sets"][bset]["path"]
    bdd = os.path.join(bp["base_path"], p)
    bff = bp["sets"][bset]["base_features_filename"]
    blf = bp["sets"][bset]["base_labels_filename"]

    return bdd, bff, blf


def make_m_parameters(model, learning_rate=None, add_suffix=True):
    '''
    Args:
      * model: The model label (e.g. 'vgg1')
      * learning_rate: Optional learning rate
      * no_suffix: Don't append a suffix to the model filename 

    Returns:
      * mbp: Model base path
      * mfn: Model filename
      * mfp: Model file path name i.e. full path and name
      * create_model: model creation function
      * mdt: Model data type
    '''
    b = MODELS[model]
    mbp = MODELS['base_path']

    mfn = b['filename']
    mfp = os.path.join(mbp, mfn)
    if learning_rate:
        mfp = f'{mfp}_l{learning_rate}'
    if add_suffix and b['filename_suffix']:
        mfp = f'{mfp}{b["filename_suffix"]}'

    create_model = b['create_model']
    mdt = b['model_data']

    return mbp, mfn, mfp, create_model, mdt

def list_models_and_creator(model_label):
    mbp, mfn, mfp, model_creator, mdt = make_m_parameters(model_label, add_suffix=False)

    model_files = sorted(glob.glob(f'{mfp}*'))
    return model_files, model_creator


def my_load_data(model_data="raw22"):
    '''Loads features and labels into a "data" dict which follows the format:
    data: {
        'train': [
            (features_0.npy, labels_0.npy),
            (features_1.npy, labels_1.npy),
            ...
        ],
        'validate': [
            (features_0.npy, labels_0.npy),
        ]
        'test': [
            (features_0.npy, labels_0.npy),
        ]
    }

    '''
    data = defaultdict(dict)
    for bset in BUNDLED_DATA[model_data]['sets'].keys():
        bdd, bff, blf = make_b_parameters(model_data, bset)
        features_files = sorted(glob.glob(f'{bdd}/{bff}_*.npy'))
        labels_files = sorted(glob.glob(f'{bdd}/{blf}_*.npy'))
        data[bset] = []
        for i, features_file in enumerate(features_files):
            labels_file = labels_files[i]
            features_file = os.path.join(bdd, features_file)
            labels_file = os.path.join(bdd, labels_file)
            data[bset].append((i, features_file, labels_file))
    
    return data


def make_set_filename(bdd, base_filename, set_count):
    return os.path.join(bdd, f'{base_filename}_{set_count}.npy')


def save_features_and_labels(bdd, bff, blf, features, labels, set_count):
    f = np.array(features)   # pylint: disable=invalid-name
    filename = make_set_filename(bdd, bff, set_count)
    print(f'Save set {set_count}')
    np.save(filename, f)

    # Save and reset labels
    f = np.array(labels)   # pylint: disable=invalid-name
    filename = make_set_filename(bdd, blf, set_count)
    np.save(filename, f)


def my_split_data(model_data="vgg22"):
    '''Split a single large numpy saved file file into smaller files.'''

    print(f'Split data {model_data}')

    data = my_load_data(model_data)
    features = np.array([])
    labels = np.array([])
    for i, features_file, labels_file in data['train']:
        print(f'Loading set {i} of features/labels data.')
        features = np.load(features_file, mmap_mode='r')
        labels = np.load(labels_file, mmap_mode='r')
        # features = np.append(features, f)
        # labels = np.append(labels, l)

    bdd, bff, blf = make_b_parameters(model_data, 'train')
    print(features.shape, labels.shape)
    shape = features.shape

    increments = math.ceil(shape[0]/38)
    for i in range(0, 38):
        i_start = i * increments
        i_end = i_start + increments
        print(i_start, i_end)
        save_features_and_labels(bdd, f'_{bff}', f'_{blf}', features[i_start:i_end],
                                labels[i_start:i_end], i)


def load_data(vector_length=128):
    """A function to load gender recognition dataset from `data` folder
    After the second run, this will load from results/features.npy and
    results/labels.npy files as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them
    # from there instead
    if (os.path.isfile("results/features.npy") and
            os.path.isfile("results/labels.npy")):
        features = np.load("results/features.npy")
        labels = np.load("results/labels.npy")
        return features, labels
    # read dataframe
    df = pd.read_csv("balanced-all.csv")
    # get total samples
    n_samples = len(df)
    # get total male samples
    n_male_samples = len(df[df["gender"] == "male"])
    # get total female samples
    n_female_samples = len(df[df["gender"] == "female"])
    print("Total samples:", n_samples)
    print("Total male samples:", n_male_samples)
    print("Total female samples:", n_female_samples)
    # initialize an empty array for all audio features
    X = np.zeros((n_samples, vector_length))
    # initialize an empty array for all audio labels
    # (1 for male and 0 for female)
    y = np.zeros((n_samples, 1))
    for i, (filename, gender) in tqdm.tqdm(
        enumerate(zip(df["filename"], df["gender"])), "Loading data",
        total=n_samples
    ):
        features = np.load(filename)
        X[i] = features
        y[i] = LABEL2INT[gender]
    # save the audio features and labels into files
    # so we won't load each one of them next run
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y


def split_data(X, y, test_size=0.1, valid_size=0.1):
    # split training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=7
    )
    # split training set and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=7
    )
    # return a dictionary of values
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def get_model_creator(label, vector_length=22050):
    """Create the ML model based on the ML model label."""
    return MODELS[label]["create_model"]



def process_sources():
    # Process the original data set(s)
    source = 'fl'
    osets = ORIGINAL_DATA[source]['sets'].keys()
    for oset in osets:
        print(f'Processing source {source} oset {oset}')
        sdd, sdm = make_o_parameters(source, oset)
        print(sdd, sdm)
        task_list = []
        if source == 'cv':
            task_list = cv_processor.cv_task_list(sdd, sdm)
        elif source == 'fl':
            task_list = cv_processor.fl_task_list(sdd, sdm)

        pdd, pdm = make_p_parameters('raw22', oset)
        cvp = cv_processor.CommonVoiceProcessor(task_list, pdd, pdm, f'{oset}_')
        cvp.generate_raw22(processes_count=2)


def process_bundles():
    # # And bundle the processed datasets
    format = 'raw22'
    bsets = BUNDLED_DATA[format]['sets'].keys()

    for bset in bsets:
        print(f'Processing format {format} bset {bset}')
        pdd, pdm = make_p_parameters(format, bset)
        bdd, bff, blf = make_b_parameters(format, bset)
        cv_processor.generate_features_and_labels_npy(pdd, pdm, bdd, bff, blf,
                                        max_set_size=cv_processor.MAX_SET_SIZE)

def process2_bundles():
    # # And bundle the processed datasets
    format_in = 'raw22'
    format_out = 'vgg22'
    bsets = BUNDLED_DATA[format_in]['sets'].keys()

    for bset in bsets:
        print(f'Processing format {format_in} bset {bset}')
        pdd, pdm = make_p_parameters(format_in, bset)
        print(pdm)
        p2db = make_p2_parameters(format_out, bset)

        cvp = cv_processor.VGG22_Processor(pdd, pdm, p2db)
        cvp.process()

def main():
    # process2_bundles()
    # my_split_data(model_data='vgg22')
    print(list_models('vgg2'))


if __name__ == '__main__':
    main()