import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.io as sio
import scipy.optimize
import tqdm

if __name__ == "__main__":
    traces = ['090711e_0006', '090811c_0002', '090811d_0002', '090811d_0004','091111a_0001',
          '091111a_0003','091111c_0003','091211a_0002','091211a_0005']

    if not os.path.isdir('fit'):
        os.makedirs('fit')
    for i in tqdm.trange(len(traces)), desc='Fish':
        saccade_file = sio.loadmat('../../../data/active/fixed/saccades/'+traces[i]+'.mat')
        eye_data = saccade_file['data'][0];
        trange_fix = saccade_file['trange'][0]

        fits, lls, sses = fitting_functions.fitNEyePositions(trange_fix, np.reshape(eye_data/eye_data[0], (1, len(eye_data))), max_num_components=3,
                                                                      num_ics=100, inverse_tau_max = 1/(3*2e-3))
        best_traces = np.argmax(lls, axis=1)
        best_num_components = np.argmax(np.max(lls, axis=1));
        best_fit = fits[best_num_components,0][best_traces[best_num_components],:]

        sio.savemat('fit/'+traces[i]+'.mat',
           {'trange':trange_fix, 'fit':best_fit, 'data':eye_data,
            'model':fitting_functions.exponentialModel(trange_fix, best_fit)})
