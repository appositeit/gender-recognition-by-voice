'''Train the model on the prepared training data.'''

# Uncomment to root cause segfaults
#  import faulthandler
#  faulthandler.enable()

import gc
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] ="2"
import math
import numpy as np

import wandb
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import tensorflow as tf
import tensorflow.compat.v1 as tf1
#  tf.config.experimental.set_lms_enabled(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
import tf_slim as slim

import vggish_input
import vggish_params
import vggish_slim
import training_data
# from training_data import split_data, create_model, my_load_data
import model_functions

# from random import shuffle

# import tensorflow.compat.v1 as tf

flags = tf1.app.flags

flags.DEFINE_integer(
    'num_batches', 30,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 1

MODEL_LABEL = 'vgg1'

VECTOR_LENGTH = 22050

def run_trainer(model_label, learning_rate):
    print(f'\n\nTraining {model_label} at learning_rate {learning_rate}.')
    wandb.init(project="genderml")
    mbp, mfn, mfp, model_creator, mdt = training_data.make_m_parameters(model_label, learning_rate=learning_rate)

    # construct the model
    if os.path.isfile(mfp):
        print('Loading existing model.')
        model = load_model(mfp)
    else:
        print('Creating model.')
        model = model_creator(model_label=model_label)

    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    # print summary of the model
    model.summary()
    # load the dataset
    print('Load dataset')

    data = training_data.my_load_data(model_data=mdt)
    #  model = create_model(vector_length=VECTOR_LENGTH)
    # use tensorboard to view metrics
    tensorboard = TensorBoard(log_dir="logs")
    # define early stopping to stop training after 5 epochs of not improving
    early_stopping = EarlyStopping(mode="min", patience=40,
                                restore_best_weights=True)

    BATCH_SIZE = 100
    EPOCHS = 800

    wandb.config.update = ({
        'learning_rate': learning_rate,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'model_label' : model_label,
        # 'allow_val_change'=True,
    })
    run_name = wandb.run.name
    mfp2 = os.path.join(mbp, 'by_run', f'{run_name}.h5')
    print(f'run_name: {run_name}')

    print('Train model')
    # train the model using the training set and validating using validation set
    # We have multiple training data sets, so we loop over the training feature/label sets
    # during our training.
    # We progressively load the data sets into memory because the data can be bigger than memory.
    _, features_file, labels_file = data['validate'][0]
    validate_features = np.load(features_file, allow_pickle=True)
    validate_labels = np.load(labels_file, allow_pickle=True)

    for i, features_file, labels_file in data['train']:
        print(f'training on set {i} of features/labels data.')
        features = np.load(features_file, allow_pickle=True)
        labels = np.load(labels_file, allow_pickle=True)
        model.fit(features, labels,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(validate_features, validate_labels),
                callbacks=[tensorboard, early_stopping, WandbMetricsLogger()])
        model.save(mfp)
        model.save(mfp2)
        gc.collect()

    print('Save model')
    # save the model to a file

    # evaluating the model using the testing set
    print(f"Evaluating the model using {len(data['X_test'])} samples...")
    _, features_file, labels_file = data['test'][0]
    test_features = np.load(features_file, allow_pickle=True)
    test_labels = np.load(labels_file, allow_pickle=True)
    loss, accuracy = model.evaluate(test_features, test_labels, verbose=0)
    wandb.log({"loss": loss, "accuracy": accuracy})
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    wandb.finish()


def describe_variable(name, v):
    print(f'{name}: {type(v)} {v.shape} {tf.shape(v)}')

def main(_):
    # Run non-VGGish model trainer
    # for model_label in ['6layer', '5layer', '4layer']:
    for model_label in ['vgg2']:
        for learning_rate in [0.0001, 0.0001, 0.0001]:
            run_trainer(model_label, learning_rate)

    # RUn VGGish model trainer
    # for model_label in ['vgg1']:
    #     # for learning_rate in [0.01, 0.005, 0.001]:
    #     for learning_rate in [0.01]:
    #         run_vggish_trainer(model_label, learning_rate)


# """
# Usage:
#   # Run training for 100 steps using a model checkpoint in the default
#   # location (vggish_model.ckpt in the current directory). Allow VGGish
#   # to get fine-tuned.
#   $ python vggish_train_demo.py --num_batches 100

#   # Same as before but run for fewer steps and don't change VGGish parameters
#   # and use a checkpoint in a different location
#   $ python vggish_train_demo.py --num_batches 50 \
#                                 --train_vggish=False \
#                                 --checkpoint /path/to/model/checkpoint
# """
#
# def _get_examples_batch():
#   """Returns a shuffled batch of examples of all audio classes.

#   Note that this is just a toy function because this is a simple demo intended
#   to illustrate how the training code might work.

#   Returns:
#     a tuple (features, labels) where features is a NumPy array of shape
#     [batch_size, num_frames, num_bands] where the batch_size is variable and
#     each row is a log mel spectrogram patch of shape [num_frames, num_bands]
#     suitable for feeding VGGish, while labels is a NumPy array of shape
#     [batch_size, num_classes] where each row is a multi-hot label vector that
#     provides the labels for corresponding rows in features.
#   """
#   # Make a waveform for each class.
#   num_seconds = 5
#   sr = 44100  # Sampling rate.
#   t = np.arange(0, num_seconds, 1 / sr)  # Time axis
#   # Random sine wave.
#   freq = np.random.uniform(100, 1000)
#   sine = np.sin(2 * np.pi * freq * t)
#   # Random constant signal.
#   magnitude = np.random.uniform(-1, 1)
#   const = magnitude * t
#   # White noise.
#   noise = np.random.normal(-1, 1, size=t.shape)

#   # Make examples of each signal and corresponding labels.
#   # Sine is class index 0, Const class index 1, Noise class index 2.
#   sine_examples = vggish_input.waveform_to_examples(sine, sr)
#   sine_labels = np.array([[1, 0, 0]] * sine_examples.shape[0])
#   const_examples = vggish_input.waveform_to_examples(const, sr)
#   const_labels = np.array([[0, 1, 0]] * const_examples.shape[0])
#   noise_examples = vggish_input.waveform_to_examples(noise, sr)
#   noise_labels = np.array([[0, 0, 1]] * noise_examples.shape[0])

#   # Shuffle (example, label) pairs across all classes.
#   all_examples = np.concatenate((sine_examples, const_examples, noise_examples))
#   all_labels = np.concatenate((sine_labels, const_labels, noise_labels))
#   labeled_examples = list(zip(all_examples, all_labels))
#   shuffle(labeled_examples)

#   # Separate and return the features and labels.
#   features = [example for (example, _) in labeled_examples]
#   labels = [label for (_, label) in labeled_examples]
#   return (features, labels)



if __name__ == '__main__':
    tf1.app.run()