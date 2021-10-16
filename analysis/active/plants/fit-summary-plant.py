import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.io as sio
import scipy.optimize
import tqdm

num_ics = 100
if __name__ == "__main__":
    traces = [ '090711e_0006', '090811c_0002','090811d_0002','090811d_0004',
                '091111a_0001', '091111a_0003', '091111c_0003', '091211a_0002', '091211a_0005']
    best_num_components_extrap = {'090711e_0006':2, '090811c_0002':2, '090811d_0002':2, '090811d_0004':2,'091111a_0001':2,
                        '091111a_0003':3,'091111c_0003':2,'091211a_0002':2,'091211a_0005':2}

    if not os.path.isdir('summary-plant'):
        os.makedirs('summary-plant')

    sr_data = [[] for i in range(len(traces))]
    sr_tranges = [[] for i in range(len(traces))]
    # release_indices = [[] for i in range(len(traces))]

    for i in range(len(traces)):
        trange, eye_pos, pe_start_index, displacement_index, release_index, step_pos = fitting_functions.importActiveData('../../../data/active/fixed/'+traces[i]+'.mat')

        n = best_num_components_extrap[traces[i]]
        fit_file = sio.loadmat('../extrap/results/'+traces[i]+'.mat')
        lls = fit_file['lls']
        fits = fit_file['fits']
        best_fit_ind = np.argmax(lls[n-1,:])

        # generate extrapolation
        extrap_best = fitting_functions.exponentialModel(trange, fits[n-1,0][best_fit_ind,:])*eye_pos[0]
        sr_data[i] = extrap_best[release_index:] - eye_pos[release_index:]
        sr_data[i] /= sr_data[i][0]

        sr_tranges[i] = trange[release_index:] - trange[release_index]

    # Fit initial extrapolation
    fits, lls, sses = fitting_functions.fitNEyePositions(sr_tranges, sr_data, max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
    # Save initial extrapolations
    sio.savemat('summary-plant/sr_fits.mat', {'fits': fits, 'sses':sses, 'lls':lls})
