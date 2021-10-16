import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

import os

if __name__ == "__main__":
    num_ics = 100
    if len(sys.argv) == 2:
        num_ics = int(sys.argv[1])
    ms222_traces = ['091311a', '091311b', '091311c', '091311d', '091311e', '091311f', '091411a', '091411d', '091411e', '091411f']

    if not os.path.isdir('results'):
        os.makedirs('results')
    if not os.path.isdir('results/MS-222'):
        os.makedirs('results/MS-222')

    for fish_num in tqdm.trange(len(ms222_traces), desc='Trace no.:'):
        trange, pe_short, pe_long = fitting_functions.importDataMS222('../../../data/anesthetized/fixed/MS-222/'+ms222_traces[fish_num]+'.mat')
        fits, lls, sses = fitting_functions.fitNEyePositions(trange, np.vstack((pe_short, pe_long)), max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
        sio.savemat('results/MS-222/'+ms222_traces[fish_num]+'.mat', {'fits': fits, 'sses':sses, 'lls':lls}, appendmat=False)
