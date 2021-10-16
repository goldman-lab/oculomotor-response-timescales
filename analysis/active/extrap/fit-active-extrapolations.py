import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

if __name__ == "__main__":
    traces = ['090711e_0006', '090811c_0002', '090811d_0002', '090811d_0004','091111a_0001',
          '091111a_0003','091111c_0003','091211a_0002','091211a_0005']
    num_ics = 100

    for fish_num in tqdm.trange(len(traces), desc='Trace no.'):
        trange, eye_pos, pe_start_index, displacement_index, release_index, step_pos = fitting_functions.importActiveData('../../../data/active/fixed/'+traces[fish_num]+'.mat')
        # Fit initial extrapolation
        fits_original, lls_original, sses_original = fitting_functions.fitNEyePositions(trange[:displacement_index], np.reshape(eye_pos[:displacement_index]/eye_pos[0], (1, displacement_index)), max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
        # Save initial extrapolations
        sio.savemat('results/'+traces[fish_num]+'.mat', {'fits': fits_original, 'sses':sses_original, 'lls':lls_original})
