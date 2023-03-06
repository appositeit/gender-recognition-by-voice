'''Testing.'''
import unittest

import training_data


class TestMakeParameters(unittest.TestCase):
    '''Testing.'''

    def test_make_o_parameters(self):
        '''Test make_o_paramaters.'''
        base = '/mnt/working/jem/ml/source/common_voice'
        expected = (f'{base}/cv-valid-train/cv-valid-train',
                    f'{base}/cv-valid-train.csv')
        got = training_data.make_o_parameters('cv', 'train')
        self.assertEqual(expected, got)

    def test_make_p_parameters(self):
        '''Test make_p_parameters.'''
        base = '/mnt/fastest/jem/ml/training_data/processed'
        expected = (f'{base}/train',
                    f'{base}/train.sqlite')
        got = training_data.make_p_parameters('raw22', 'train')
        self.assertEqual(expected, got)


if __name__ == '__main__':
    unittest.main()
