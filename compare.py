from pickle import load
import numpy as np
from os import listdir, mkdir
from os.path import isdir

TEST_RES_PATH = r"data\test\res.pkl"
OUTPUT_RES_PATH = r'outputs'

if not isdir('outputs'):
    mkdir('outputs')


def gini_impurity(x):
    """ Gini impurity of vector (only for 2 classes) """
    if x.size == 0:
        return 0
    p_ones = x[x == 1].size / x.size
    return 2*(p_ones - (p_ones ** 2))

def calc_impurity(test_res, guess_res):
    data_n = test_res.size
    class_0 = test_res[guess_res == 0]
    class_1 = test_res[guess_res == 1]

    class_0_impurity = gini_impurity(class_0)
    class_1_impurity = gini_impurity(class_1)

    return (class_0.size * class_0_impurity + class_1.size * class_1_impurity) / data_n


def calc_score(test_res, guess_res):
    """ compares test results to true results """
    if type(test_res) is not dict:
        if type(test_res[0]) is dict:
            score = []
            for cur_guess in guess_res:
                score.append(1-calc_impurity(test_res=test_res, guess_res=cur_guess))
        else:
            return 1-calc_impurity(test_res=test_res, guess_res=guess_res)
    else:
        score = []
        for key, cur_test_res in test_res.items():
            cur_guess_res = guess_res[key]
            score.append(1 - calc_impurity(test_res=cur_test_res, guess_res=cur_guess_res))

    return np.mean(score)


def compare_dir(output_dir=OUTPUT_RES_PATH, test_res_path=TEST_RES_PATH):
    """ compares directory to test results """
    with open(test_res_path, 'rb') as ff:
        true_res = load(ff)
    scores = {}
    for res_path in listdir(output_dir):
        if res_path.endswith('.pkl'):
            try:
                with open(output_dir +'\\' + res_path, 'rb') as ff:
                    cur_res = load(ff)
                scores[res_path] = calc_score(true_res, cur_res)
            except:
                scores[res_path] = 'FAILED'
            print('{0}:   {1}'.format(res_path, scores[res_path]))

    with open(r'outputs\scores.csv', 'w') as ff:
        for key, val in scores.items():
            ff.write('{0},{1}\n'.format(key, val))


def compare_pdws(guess_pdws, test_pdws):
    if 'CLASS' in guess_pdws and 'CLASS' in test_pdws:
        return calc_score(test_res=test_pdws['CLASS'], guess_res=guess_pdws['CLASS'])
    else:
        raise Exception('unlabeled pdws')


if __name__ == '__main__':
    compare_dir()
