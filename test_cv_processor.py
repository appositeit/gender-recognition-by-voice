
'''Testing.'''
import os
import glob
import unittest
import re

import training_data
import cv_processor
import data_schema


class TestTaskList(unittest.TestCase):
    '''Testing.'''

    def test_task_list(self):
        '''Test task_list.'''
        sdd, sdm = training_data.make_o_parameters('cv', 'train')

        tasks = cv_processor.task_list(sdd, sdm)
        self.assertTrue(len(tasks) > 0)


class TestGenerateOffsets(unittest.TestCase):
    '''Testing.'''

    def test_generate_offsets(self):
        '''Test generate_offsets.'''
        func = cv_processor.generate_offsets
        self.assertEqual(func(800, 1000, 300), [])
        self.assertEqual(func(1200, 1000, 300), [0])
        self.assertEqual(func(1400, 1000, 300), [0, 300])
        self.assertEqual(func(1601, 1000, 300), [0, 300, 600])


class TestMakeProcessedFilenames(unittest.TestCase):
    '''testing.'''

    def test_make_processed_filenamess(self):
        '''test generate_offsets.'''
        func = cv_processor.make_processed_filenames
        # (filename, offsets, prefix='', suffix=''):
        self.assertEqual(func('sample.wav', []), [])
        self.assertEqual(func('sample.wav', [0]), ['sample_o00000ms.wav'])
        self.assertEqual(func('sample.wav', [0, 300], new_ext='.mp3'), ['sample_o00000ms.mp3', 'sample_o00300ms.mp3'])
        self.assertEqual(func('sample.wav', [0, 300, 600], prefix='cv_'), ['cv_sample_o00000ms.wav', 'cv_sample_o00300ms.wav', 'cv_sample_o00600ms.wav'])


def clean_directory(path):
    files = glob.glob(f'{path}/*')
    print(files)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


class TestProcessSourceFile(unittest.TestCase):
    '''testing.'''

    def test_process_source_file(self):
        '''test process_source_file.''' 
        func = cv_processor.process_source_file
        base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'code_test')
        sdd = os.path.join(base, 'source')
        sdm = os.path.join(base, 'source.csv')
        pdd = os.path.join(base, 'processed')
        pdm = os.path.join(base, 'processed.sqlite')
        prefix = 'cv_'

        clean_directory(pdd)

        gdb = data_schema.GenderDB(pdm)

        source_file = os.path.join(sdd, 'sample-000003.mp3')
        gender = 'male'
        func(gdb, pdd, source_file, gender, 'cv_')
       

# class TestGenerateFeatures(unittest.TestCase):
#     '''testing.'''

#     def test_generate_features_npy(self):
#         '''test process_source_file.''' 
#         func = cv_processor.generate_features_npy
#         base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'code_test')
#         pdd = os.path.join(base, 'processed')
#         pdm = os.path.join(base, 'processed.sqlite')
#         bdd = os.path.join(base, 'bundled')
#         bdm = 'bundled'

#         clean_directory(bdd)

#         func(pdd, pdm, bdd, bdm, max_set_size=10000)


class TestFLProcessing(unittest.TestCase):
    '''testing.'''

    def test_decode_fl_names(self):
        '''test process_source_file.''' 
        func = cv_processor.decode_fl_names
        names = [
            'de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment10.flac',
            'en_m_011f3a2d0aa2880305c08b76873c3e10.fragment10.speed2.flac',
        ]
        for name in names:
            r = func(name)
            print(name, r)


# class TestGenerateRaw22(unittest.TestCase):
#     '''Testing.'''

    # def test_generate_raw22(self):
    #     '''Test generate_offsets.'''
    #     func = cv_processor.generate_raw22
    #     base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'code_test')
    #     sdd = os.path.join(base, 'source')
    #     sdm = os.path.join(base, 'source.csv')
    #     pdd = os.path.join(base, 'processed')
    #     pdm = os.path.join(base, 'processed.sqlite')
    #     prefix = 'cv_'
       
    #     output = func(sdd, sdm, pdd, pdm, 'cv_', processes_count=1)


if __name__ == '__main__':
    unittest.main()
