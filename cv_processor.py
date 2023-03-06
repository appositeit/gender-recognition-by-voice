#!/home/jem/miniconda3/envs/gr/bin/python
'''Process source data to make training data.

For details of the training data format and structures see the training_data.py
file.

This module focuses on the active code necessary to process source data
(such as Mozilla's Common Voice) into processed and bundled data. Note
that both feature (sound sample or derived propertie) and label (gender) data
need to be prerved in this process.

Processed data is simply numpy feature and label vectors in arrays ready for
fast loading during training,

## Processing

We want to modify some or all of the samples to remove bias and make the model
robust against noise. In particular:G
    * Frequency shift male and female voices into the androgonous zone so the
    model isn't just basing it's decisions on pitch.
    * Add some common types of noise so the model can account for bad
    microphone setups.
'''
from collections import defaultdict
from multiprocessing import Process, Queue, Manager
import math
import glob
import os
import queue  # imported for using queue.Empty exception
import sys
import time
from pathlib import Path
import re
import warnings
warnings.simplefilter(action='ignore')
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

import crepe
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment as am
import soundfile as sf
import tqdm
import vggish_input


import data_schema


#  if not sys.warnoptions:
    #  import warnings
    #  warnings.simplefilter("ignore")



#  import tensorflow as tf
#  tf.get_logger().setLevel('INFO')

# import utils

#  COMMON_VOICE = '/mnt/working/jem/ml/common_voice'
#  COMMON_VOICE = '/mnt/ml/common_voice'
COMMON_VOICE = '/mnt/fastest/jem/ml/common_voice'

SAMPLE_LENGTH = 1000  # Sample length in ms
SAMPLE_OFFSET = 300  # We subsample every SAMPLE_OFFSET ms intervals

# A 2GB in memory file makes a roughly 8.1GB on disk file.
GB = 2**30   # Nuber of bytes in a Gigabyte
MAX_SET_SIZE = 1*GB     # Approx bundled file size per set in bytes. This represents in memory usage

CV_SOURCE_SAMPLE_RATE = 44100
PROCESSED_SAMPLE_RATE = 22050 

DATASETS = [
    'cv-valid-dev',
    'cv-valid-test',
    'cv-valid-train',
]
DATASET = DATASETS[0]

PROCESSES_COUNT = 8

DB_BATCH_SIZE = 100


LABEL2INT = {
    "male": 1,
    "female": 0,
    "m": 1,
    "f": 0,
}


# de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment10.flac

def decode_fl_names(filename):
    prog = re.compile('.*\/(?P<language>\w\w)_(?P<gender>\w)_(?P<name>.*)\.fragment(?P<id>\d+)\.?(?P<effect>.*)\.flac')
    m = prog.match(filename)
    if m:
        return {
            'language': m.group('language'),
            'name': m.group('name'),
            'gender': m.group('gender'),
            'id': m.group('id'),
            'effect': m.group('effect'),
        }
    return {}




def make_processed_filenames(source_filename, offsets, prefix='', suffix='',
                             new_ext=''):
    '''Generate the filename for the cache audio files.'''
    _, base = os.path.split(source_filename)
    name, ext = os.path.splitext(base)
    if new_ext:
        ext = new_ext
    names = []
    for offset in offsets:
        new_name = f'{prefix}{name}_o{offset:05}ms{suffix}{ext}'
        names.append(new_name)
    return names


def generate_offsets(original_length, sample_length, sample_offset):
    '''
    All timings should be in consistent units- typically milliseconds.
    '''
    offset_count = math.ceil((original_length - sample_length + 1)
                             / sample_offset)
    offsets = [i * sample_offset for i in range(0, offset_count)]
    return offsets


def clean_directory(path):
    files = glob.glob(f'{path}/*')

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def make_set_filename(bdd, base_filename, set_count):
    return os.path.join(bdd, f'{base_filename}_{set_count}.npy')

def generate_features_and_labels_npy(pdd, pdm, bdd, bff, blf,
                          max_set_size=MAX_SET_SIZE):
    '''Regenerate the features.npy cache.'''
    if not os.path.isdir(bdd):
        os.mkdir(bdd)

    clean_directory(bdd)    

    gdb = data_schema.GenderDB(pdm)
    files = gdb.read_valid_samples()

    features = []
    labels = []
    set_size = 0
    set_count = 0
    for f in tqdm.tqdm(files,   # pylint: disable=invalid-name
                       'Loading audio', total=len(files)):
        f_path = os.path.join(pdd, f['filename'])
        audio, _ = librosa.load(f_path)
        features.append(audio)
        labels.append(f['male'])

        set_size += len(audio)
        if set_size >= max_set_size:
            # Save and reset features
            f = np.array(features)   # pylint: disable=invalid-name
            filename = make_set_filename(bdd, bff, set_count)
            set_size = 0
            print(f'Save set {set_size}')
            np.save(filename, f)
            features = []

            # Save and reset labels
            f = np.array(labels)   # pylint: disable=invalid-name
            filename = make_set_filename(bdd, blf, set_count)
            np.save(filename, f)
            labels = []

            set_count += 1

    if features:
        # Save features
        f = np.array(features)   # pylint: disable=invalid-name
        filename = make_set_filename(bdd, bff, set_count)
        np.save(filename, f)

        # Save labels
        f = np.array(labels)   # pylint: disable=invalid-name
        filename = make_set_filename(bdd, blf, set_count)
        np.save(filename, f)


# def generate_labels_npy(  # pylint: disable=too-many-locals
#         dataset='cv-valid-train', common_voice=COMMON_VOICE,
#         max_set_size=MAX_SET_SIZE):
#     '''Regenerate the labels.npy cache.'''

#     results_path = os.path.join(common_voice, dataset, 'results')
#     cache_path = os.path.join(results_path, 'cache')
#     labels_path = os.path.join(results_path, "labels.npy")

#     files = os.listdir(cache_path)
#     labels = []
#     count = -1
#     for f in tqdm.tqdm(files,   # pylint: disable=invalid-name
#                        'Loading audio', total=len(files)):
#         _, base = os.path.split(f)
#         name, _ = os.path.splitext(base)
#         labels.append(filename_lookup[name.split('_')[-1]])
#         count += 0
#         #  if count >= max_set_size:
#         #      break

#     if labels:
#         np.save(labels_path, np.array(labels))

def cv_task_list(sdd, sdm):
    '''List source files.

    Pushes a tuple off:
    * Source file pathname.
    * Gender.
    '''

    #  # if features & labels already loaded individually and bundled,
    # load them from there instead
    #  features_path = os.path.join(results_path, "features.npy")
    #  labels_path = os.path.join(results_path, "labels.npy")
    #  if os.path.isfile(features_path) and os.path.isfile(labels_path):
    #  features = np.load(features_path)
    #  labels = np.load(labels_path)
    #  return features, labels

    df = pd.read_csv(sdm)   # pylint: disable=invalid-name
    # get total samples
    n_samples = len(df)
    # get total male samples
    n_male_samples = len(df[df['gender'] == 'male'])
    # get total female samples
    n_female_samples = len(df[df['gender'] == 'female'])
    # print("Total samples:", n_samples)
    # print("Total male samples:", n_male_samples)
    # print("Total female samples:", n_female_samples)

    tasks = []
    for d in df.itertuples():   # pylint: disable=invalid-name
        if str(d.gender) == 'nan' or d.gender == 'other':
            continue
        filename = os.path.split(d.filename)[1]
        source_filepathname = os.path.join(sdd, filename)
        tasks.append((source_filepathname, d.gender))

    return tasks


def fl_task_list(sdd, _):
    '''List source files.

    Pushes a tuple off:
    * Source file pathname.
    * Gender.
    '''

    tasks = []
    files = glob.glob(f'{sdd}/*.flac')
    for f in files:
        r = decode_fl_names(f)
        if r['effect']:
            continue
        source_filepathname = os.path.join(sdd, f)
        tasks.append((source_filepathname, r['gender']))
    
    return tasks


class CommonVoiceProcessor:
    '''Processess common voice from mozilla files.

    Runs a parallel threaded generator to process source to processed data.

    Inputs:
    sdd, sdm, pdd, pdm: As described elsewhere.
    pprefix: A prefix to put on the name of the processed files.
    '''

    def __init__(self, task_list, pdd, pdm, pprefix):
        self.tasks = task_list
        self.pdd = pdd
        self.pdm = pdm
        self.pprefix = pprefix
        self.total_count = len(self.tasks)

        self.tasks_todo = Queue()
        self.tasks_done = Queue()

    def generate_raw22(self, processes_count=PROCESSES_COUNT):
        '''
        Runs a parallel threaded generator to process source to processed data.

        Inputs:
        sdd, sdm, pdd, pdm: As described elsewhere.
        pprefix: A prefix to put on the name of the processed files.
        processes_count: Limit the number of parallel processes being run.
        '''

        # If the destination directory doesn't exist, create it.
        if not os.path.isdir(self.pdd):
            os.mkdir(self.pdd)

        # processes = []
        # print(f'All tasks submitted: {self.total_count}')

        # Creating processes
        with Manager() as manager:

            lock = manager.Lock()
            tasks_todo = manager.Queue()
            for task in self.tasks:
                tasks_todo.put(task, block=False)
            tasks_done = manager.Queue()

            pool = []
            for _ in range(processes_count):
                gdb = data_schema.GenderDB(self.pdm)
                pool.append(Process(target=self.run_process, args=(gdb, self.pdd, self.pprefix, tasks_todo, tasks_done)))

            for p in pool:
                p.start()


        # for w in range(0, processes_count):
        #     p = Process(target=self.run_process, args=(gdb, self.pdd, self.pprefix))
        #     processes.append(p)
        #     time.sleep(0.1)
        #     p.start()
            print('All processes activated')

            time.sleep(1)
            with tqdm.tqdm(total=self.total_count, desc='Processing source data:') as pbar:
                while (not tasks_todo.empty()) or (not self.tasks_done.empty()):
                    #  print(tasks_todo.qsize(), tasks_done.qsize())
                    try:
                        done = tasks_done.get(block=False)
                        if done:
                            pbar.update(1)
                    except:
                        pass

            print('Waiting for processes to complete.')
            for p in pool:
                p.join()
        # for p in processes:
        #     p.join()

        print('Done.')

        return True

    def run_process(self, gdb, pdd, pprefix, tasks_todo, tasks_done):
        queue_strikes = 0
        while (not tasks_todo.empty()):
            try:
                '''
                    try to get task from the queue. get_nowait() function will
                    raise queue.empty exception if the queue is empty.
                    queue(false) function would do the same task also.
                '''
                source_file, gender = tasks_todo.get(True)
                process_source_file(gdb, pdd, source_file, gender, pprefix)
                tasks_done.put(source_file)

            except: # Queue.empty:
                pass
                #  queue_strikes += 1
                #  print(f'queue empty {queue_strikes} times.')
                #  if queue_strikes > 10:
                    #  break
            else:
                '''
                    if no exception has been raised, add the task completion
                    message to task_that_are_done queue
                '''
                #  print(task)
                #  tasks_done.put(task[0] + ' is done by ' + current_process().name)
                pass
        return True

    def populate_valid(self):
        '''Quick hack function. Not for normal use!'''
        filename = '/mnt/fastest/jem/ml/training_data/processed/train.sqlite'
        gdb = data_schema.GenderDB(filename)
        samples = gdb.read_samples()
        valid = True
        with tqdm.tqdm(total=len(samples), desc='Set valid files:') as pbar:
            for sample in samples:
                filename = sample['filename']
                stats = os.stat(filename)
                if stats.st_size == 0:
                    valid = False
                if not valid:
                    gdb.update_sample(sample['id'], {
                        'valid': valid,
                        })
                pbar.update(1)



def process_source_file(gdb, pdd, source_file, gender, pprefix):
    '''Splits a source file into 1s chunks.

    This process can be iterated, and we want it to be fast. So:
    * Lookup to see if we've already seen this source file.
    * If so, load file length, check for presence of split files, skip
    * Otherwise process the file.

    Inputs:
        gdb: GenderDB object.
        pdd: Destination directory for processed files.
        soure_file: Full filepath to the source file.
        gender: The gender marker for the source file.
    '''
    # If we find any samples based on the original, presume we've
    # already processed this file and continue
    source_data = gdb.read_original(source_file)
    original_id = None
    generate_files = True 
    if source_data:
        original_length = source_data[0][4]
        original_id = source_data[0][0]
        if original_length:
            offsets = generate_offsets(original_length, SAMPLE_LENGTH, SAMPLE_OFFSET)
            filenames = make_processed_filenames(source_file, offsets, new_ext='.wav', prefix=pprefix)

            for filename in filenames:
                if not os.path.isfile(filename):
                    generate_files = False

    if not generate_files:
        return

    # Load the original source file at 22Khz.
    audio, sr = librosa.load(source_file)  # pylint: disable=invalid-name
    audio = librosa.resample(audio, orig_sr=sr, target_sr=PROCESSED_SAMPLE_RATE)
    sr = PROCESSED_SAMPLE_RATE
    
    original_length = librosa.get_duration(y=audio, sr=sr) * 1000

    if not original_id:
        original_id = gdb.add_originals([{
            'filename': source_file,
            'male': LABEL2INT[gender],
            'length': original_length,
            }])
    elif not original_length:
        gdb.update_orginal(original_id, [{
            'filename': source_file,
            'male': LABEL2INT[gender],
            'length': original_length,
            }])

    if not original_id:
        return


    # Timings are all in ms at this point.
    offsets = generate_offsets(original_length, SAMPLE_LENGTH, SAMPLE_OFFSET)
    filenames = make_processed_filenames(source_file, offsets, new_ext='.wav', prefix=pprefix)

    samples = []
    for count, filename in enumerate(filenames):
        filepathname = os.path.join(pdd, filename)
        # Timings are still in ms
        t_start = offsets[count]
        t_finish = t_start + SAMPLE_LENGTH

        # Here we need to convert to samples
        a = audio[int(sr*t_start/1000):int(sr*t_finish/1000)]  # pylint: disable=invalid-name
        frame_length = int(sr*SAMPLE_LENGTH/1000)
        rms = librosa.feature.rms(y=a, frame_length=frame_length)
        rms_total = rms[0].sum()
        if rms_total < 0.1:
            # print(f'Too quiet: {filepathname}')
            Path(filepathname).touch()
            continue

        if not os.path.isfile(filepathname):
            sf.write(filepathname, a, sr, 'PCM_24')
            # librosa.output.write_wav(filepathname, a, sr, norm=False)

        samples.append({
            'filename': filepathname,
            'original_id': original_id,
            'parent_offset': t_start,
            'male': LABEL2INT[gender],
            'valid': len(a)>0,
            })

    if samples:
        gdb.add_samples(samples)

class VGG22_Processor:
    '''A multiprocessor vgg2 processor.

    Runs a parallel threaded generator to process source to bundled data.
    '''

    def __init__(self, pdd, pdm, p2db, max_set_size=MAX_SET_SIZE):
        '''Process stage 1 processed data (raw22 1s 22kHz samples) into stage 2 processed data
        which is VGGish data (essentially an array of mel spectograms). VGGish features are more
        compact and introduce useful feature analysis (particulalry the mel frequency scaling/density).

        Inputs:
            pdd, pdm: Input directory and metadata for the stage 1 processed data.
            p2db: Output directory for bundled p2 features/labels.
        '''
        self.pdd = pdd
        self.pdm = pdm
        self.p2db = p2db
        self.max_set_size = MAX_SET_SIZE

        self.base_features_filename = 'vgg_features'
        self.base_labels_filename = 'vgg_labels'

        print(p2db)
        if not os.path.isdir(p2db):
            os.mkdir(p2db)
        else:
            clean_directory(p2db)    

    def save_features_and_labels(self, p2db, features, labels, set_count):
        f = np.array(features)   # pylint: disable=invalid-name
        filename = make_set_filename(p2db, self.base_features_filename, set_count)
        print(f'Save set {set_count}')
        np.save(filename, f)

        # Save and reset labels
        f = np.array(labels)   # pylint: disable=invalid-name
        filename = make_set_filename(p2db, self.base_labels_filename, set_count)
        np.save(filename, f)

    def process(self, processes_count=PROCESSES_COUNT):
        '''
        Runs a parallel threaded generator to process source to processed data.

        Inputs:
            * processes_count: Limit the number of parallel processes being run.
        '''
        with Manager() as manager:
            tasks_done = manager.Queue()
            tasks_todo = manager.Queue()

            print('Loading tasks.')
            gdb = data_schema.GenderDB(self.pdm)
            files = gdb.read_valid_samples()
            total = 0
            for f in files:
                source_file = os.path.join(self.pdd, f['filename'])
                gender = f['male']
                tasks_todo.put((source_file, gender), block=False)
                total += 1
            gdb.close()

            pool = []
            for _ in range(processes_count):
                pool.append(Process(target=self.run_process, args=(tasks_todo, tasks_done)))

            for p in pool:
                p.start()

            print(f'All {len(pool)} processes activated')

            features = []
            labels = []
            set_size = 0
            set_count = 0
            with tqdm.tqdm(total=total, desc='Processing source data:') as pbar:
                while (not tasks_todo.empty()) or (not tasks_done.empty()):
                    #  print(tasks_todo.qsize(), tasks_done.qsize())
                    try:
                        done = tasks_done.get(block=False)
                        if done:
                            pbar.update(1)

                            features.append(done[0])
                            labels.append(done[1])

                            set_size += sys.getsizeof(done[0])
                            if set_size >= self.max_set_size:
                                # Save and reset features/labels arrays
                                self.save_features_and_labels(self.p2db, features, labels, set_count)
                                set_size = 0
                                features = []
                                labels = []
                                set_count += 1


                    except:
                        pass

            if features:
                self.save_features_and_labels(self.p2db, features, labels, set_count)

            print('Waiting for processes to complete.')
            for p in pool:
                p.join()

        print('Done.')

        return True

    def run_process(self, tasks_todo, tasks_done):
        while (not tasks_todo.empty()):
            try:
                '''
                    try to get task from the queue. get_nowait() function will
                    raise queue.empty exception if the queue is empty.
                    queue(false) function would do the same task also.
                '''
                source_file, gender = tasks_todo.get(True)
                audio, sr = librosa.load(source_file)
                vgg = vggish_input.waveform_to_examples(audio, sr)
                tasks_done.put((vgg, gender))

            except Exception as e: # Queue.empty:
                print(e)
                pass
                #  queue_strikes += 1
                #  print(f'queue empty {queue_strikes} times.')
                #  if queue_strikes > 10:
                    #  break
            else:
                '''
                    if no exception has been raised, add the task completion
                    message to task_that_are_done queue
                '''
                #  print(task)
                #  tasks_done.put(task[0] + ' is done by ' + current_process().name)
                pass
        print('Process exiting.')
        return True


def fill_pitch():
    gdb = data_schema.GenderDB()

    samples = gdb.read_samples()
    for sample in tqdm.tqdm(samples, 'Filling pitch:'):
        f = sample['frequency']
        if not ((f == None) or (f==0.0)):
            continue

        filename = sample['filename']
        audio, sr = librosa.load(filename)
        _, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=2000, verbose=0)
        #  print(f'Frequency: {frequency}, Confidence: {confidence}')

        gdb.update_sample(sample['id'], {
            'frequency': frequency,
            'confidence': confidence,
            })

def fill_melspectogram():
    gdb = data_schema.GenderDB()

    samples = gdb.read_samples()
    for sample in tqdm.tqdm(samples, 'Fill melspectogram:'):
        ms = sample['melspectogram']
        if ms:
            continue

        filename = sample['filename']
        audio, sr = librosa.load(filename)
        mel = np.mean(librosa.feature.melspectogram(audio, sr=sr).T,axis=0)
        #  result = np.hstack((result, mel))

        gdb.update_sample(sample['id'], {
            'melspectogram': mel,
            })

def generate_melspectogram_cache():
    '''Read through DB and write out the features.npy and labels.npy cache
    files.'''
    gdb = data_schema.GenderDB()
    samples = gdb.read_samples()
    features = defaultdict(list)
    labels = defaultdict(list)

    for sample in tqdm.tqdm(samples, 'Generate melspectogram'):
        dataset = sample['filename'].split('/')[6]
        feature = np.frombuffer(sample['melspectogram'], dtype=np.float32)
        features[dataset].append(feature)
        labels[dataset].append(sample['male'])

    for dataset in features.keys():
        results_path = os.path.join(COMMON_VOICE, dataset, 'results')
        features_path = os.path.join(results_path, 'features.npy')
        labels_path = os.path.join(results_path, 'labels.npy')

        np.save(features_path, np.array(features[dataset]))
        np.save(labels_path, np.array(labels[dataset]))

def main():
    '''Main.'''

    pass
    # cvp = CommonVoiceProcessor(sdd, sdm, pdd, pdm, pprefix)

    # cvp.generate_raw22(processes_count=PROCESSES_COUNT)
    #  for dataset in DATASETS:
        #  print(f'Training dataset: {dataset}')
        #  generate_processes(common_voice=COMMON_VOICE, dataset=dataset)
        #  print()

    #  # features, labels = make_samples()
    #  max_set_size = 5000000
    #  for dataset in DATASETS:
        #  print(f'Build feature and labels cache for: {dataset}')
        #  generate_features_npy(dataset=dataset, max_set_size=max_set_size)
        #  generate_labels_npy(dataset=dataset, max_set_size=max_set_size)
        #  print()

    #  print(f'Fill pitch for all samples.')
    #  fill_pitch()

    #  print(f'Fill spectogram for all samples.')
    #  fill_melspectogram()

    # print(f'Generate spectogram cache for all datasets.')
    # generate_melspectogram_cache()


if __name__ == '__main__':
    #  fill_melspectogram()
    #  fill_pitch()
    main()
