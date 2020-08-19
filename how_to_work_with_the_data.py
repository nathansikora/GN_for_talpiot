"""
    Welcome to Giluy-Naot's separation challenge.
    In this challenge you will experience algorithm development for solving problems with operational significance.

    The data is split to:   labeled training set
                            unlabeled testing set

    both of which are archived in .zip files. in \data\train and \data\test correspondingly.

    The data sets contain:  1. train.pkl / test.pkl: list of pdws buffer dictionaries (Pulse description Words):
                                        each of these is a single separate 500 msec recording of pulses from 2 origins.

                                        pdws = {
                                        'AMP':      list of pulse's measured power (amplitude)
                                        'TOA':      list of pulse's measurement times (Time Of Arrival)
                                        'CLASS':    labels (only for train)
                                        'NAME':     name of current buffer
                                        }

                                        for example, pdws['TOA'][:5] is the TOA of the first 5 pulses.
                                        pdws['AMP'][:5] is the corresponding amplitude.

                            2. Image representation of the data
                            3. res.pkl file for train labels (used for comparison)

    The score is calculated as the average Gini Impurity of the test set:
            Gini Impurity:  a measure of how often a randomly chosen element from the set would be incorrectly
                            labeled if it was randomly labeled according to the distribution of labels in the subset

    Drive link for uploading results: https://drive.google.com/drive/folders/15TREwGs882H69YqKwgIKJslN_fRNg1JJ
"""


""" Example for loading the data: """
from pickle import load
with open(r'data/train/train.pkl', 'rb') as ff:
    pdws = load(ff)

print([p['TOA'].size for p in pdws])   # amount of pulses in each buffer


""" Example for plotting the data: """
from matplotlib.pyplot import figure, plot, show
cur_pdws = pdws[0]  # using loaded pdws
figure(1)
plot(cur_pdws['TOA'], cur_pdws['AMP'], '.')
show()


""" Example for comparing pdws: """
from compare import compare_pdws
print(compare_pdws(cur_pdws, cur_pdws))     # should be 1


""" Comparing the separation result to the train/test set """
from pickle import load
from compare import calc_score
with open(r'data/train/res.pkl', 'rb') as ff:
    res = load(ff)
my_res = {p['NAME']: p['CLASS'] for p in pdws}
print(calc_score(test_res=res, guess_res=my_res))     # should be 1

