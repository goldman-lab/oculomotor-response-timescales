import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.io as sio
import scipy.optimize
import tqdm
import os

if __name__ == "__main__":
    num_ics = 100
    if len(sys.argv) == 2:
        num_ics = int(sys.argv[1])
    ketamine_traces = ['63011d','70911i', '70911l', '70911m', '82411p', '82411r']

    if not os.path.isdir('results'):
        os.makedirs('results')
    if not os.path.isdir('results/Ketamine'):
        os.makedirs('results/Ketamine')

    for fish_num in tqdm.trange(len(ketamine_traces), desc='Trace no.:'):
        trange, pe_short, pe_long = fitting_functions.importDataKetamine('../../../data/anesthetized/fixed/Ketamine/'+ketamine_traces[fish_num]+'.mat')
        fits, lls, sses = fitting_functions.fitNEyePositions(trange, np.vstack((pe_short, pe_long)), max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
        sio.savemat('results/Ketamine/'+ketamine_traces[fish_num]+'.mat', {'fits': fits, 'sses':sses, 'lls':lls}, appendmat=False)
