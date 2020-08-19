from pickle import load, dump
import numpy as np
from copy import copy
from matplotlib.pyplot import imshow, plot, cla, figure, savefig
from os import mkdir
from os.path import isdir

if not isdir('data\my_res'):
    mkdir('data\my_res')
if not isdir('outputs'):
    mkdir('outputs')

MAT_SIZE_R = 64
MAT_SIZE_C = 64
TRANS_TRESH = 5
IS_TO_SAVE_IMGS = True


def save_to_image(pdws, save_path):
    """ saves an image of pdws toa to amp plot """
    figure(1)
    cla()
    if 'CLASS' in pdws:
        for c in np.unique(pdws['CLASS']):
            cur_ii = pdws['CLASS'] == c
            plot(pdws['TOA'][cur_ii], pdws['AMP'][cur_ii], '.')
    else:
        plot(pdws['TOA'], pdws['AMP'], '.')
    savefig(save_path + '\\' + pdws['NAME'] + '.png')


def save_results(pdws, save_path=r'outputs\base_line.pkl'):
    """ saves the class assignment of pdws """
    if type(pdws) is dict:
        res = {pdws['NAME']: pdws['CLASS']}
    else:
        res = {p['NAME']: p['CLASS'] for p in pdws}
    with open(save_path, 'wb') as ff:
        dump(result, ff)
    return res


def pdws_to_matrix(pdws):
    """ converts pdws to matrix where TOA is x and AMP is y"""
    I = np.zeros((MAT_SIZE_R, MAT_SIZE_C))
    pdws = copy(pdws)

    ''' normalization '''
    pdws['AMP'] -= min(pdws['AMP'])
    pdws['AMP'] /= max(pdws['AMP'])
    pdws['TOA'] -= min(pdws['TOA'])
    pdws['TOA'] /= max(pdws['TOA'])

    ''' operation '''
    for toa, amp in zip(pdws['TOA'], pdws['AMP']):
        I[int(amp*(MAT_SIZE_R-1)), int(toa*(MAT_SIZE_C-1))] = 1
    return I


def find_pdws_in_matrix(I, pdws):
    """ finds which pdws fits in matrix """
    pdws = copy(pdws)
    in_ii = []

    ''' normalization '''
    pdws['AMP'] -= min(pdws['AMP'])
    pdws['AMP'] /= max(pdws['AMP'])
    pdws['TOA'] -= min(pdws['TOA'])
    pdws['TOA'] /= max(pdws['TOA'])

    ''' operation '''
    for ii, (toa, amp) in enumerate(zip(pdws['TOA'], pdws['AMP'])):
        if I[int(amp*(MAT_SIZE_R-1)), int(toa*(MAT_SIZE_C-1))] == 1:
            in_ii.append(ii)
    return in_ii


if __name__ == '__main__':
    """ baseline operation """
    with open(r"data\test\test.pkl", 'rb') as ff:
        data = load(ff)
    result = {}
    for pdws in data:
        test_name = pdws['NAME']
        print(test_name)
        I = pdws_to_matrix(pdws)
        pulse_assign = np.zeros(pdws['TOA'].size)

        first_I = np.zeros((MAT_SIZE_R, MAT_SIZE_C))    # first emitter
        second_I = np.zeros((MAT_SIZE_R, MAT_SIZE_C))   # second emitter

        transition = False
        assign_state = 1
        for c in range(I.shape[1]):
            cur_col = I[:, c]
            active_pxls = np.nonzero(cur_col)[0]
            if active_pxls.size == 0:
                continue
            if max(active_pxls) - min(active_pxls) < TRANS_TRESH:    # does the two emitters collide
                transition = True
            else:
                if transition:
                    assign_state *= -1
                    transition = False
                med = np.median(active_pxls)
                if assign_state == 1:
                    first_I[active_pxls[active_pxls >= med], c] = 1
                    second_I[active_pxls[active_pxls < med], c] = 1
                else:
                    first_I[active_pxls[active_pxls < med], c] = 1
                    second_I[active_pxls[active_pxls >= med], c] = 1

        pulse_assign[find_pdws_in_matrix(first_I, pdws)] = 1
        pdws['CLASS'] = pulse_assign
        if IS_TO_SAVE_IMGS:
            save_to_image(pdws, 'data\my_res')
        result[test_name] = pulse_assign


    if IS_TO_SAVE_IMGS:
        with open(r'outputs\base_line.pkl', 'wb') as ff:
            dump(result, ff)