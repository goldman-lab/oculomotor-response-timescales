import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

import os

if __name__ == "__main__":
    file_names = [('090811d_0002','090811d_0004',), ('091111a_0001', '091111a_0003'),  ('091211a_0002', '091211a_0005'),
       ('090711e_0006',), ('090811c_0002',), ('091111c_0003',)]
    num_ics = 100

    best_num_components_extrap = {'090711e_0006':2, '090811c_0002':2, '090811d_0002':2, '090811d_0004':2,'091111a_0001':2,
                        '091111a_0003':3,'091111c_0003':2,'091211a_0002':2,'091211a_0005':2}

    if not os.path.isdir('results'):
        os.makedirs('results')
    if not os.path.isdir('results/best'):
        os.makedirs('results/best')
    if not os.path.isdir('results/conservative'):
        os.makedirs('results/conservative')

    for fish_num in tqdm.trange(len(file_names), desc='Fish no.'):
        sr_data = [[] for i in range(len(file_names[fish_num]))]
        sr_tranges = [[] for i in range(len(file_names[fish_num]))]
        sr_data_conservative = [[] for i in range(len(file_names[fish_num]))]
        fish_name = file_names[fish_num][0][:-5]

        for trace_num in range(len(file_names[fish_num])):
            trange, eye_pos, pe_start_index, displacement_index, release_index, step_pos = fitting_functions.importActiveData('../../../data/active/fixed/'+file_names[fish_num][trace_num]+'.mat')

            n = best_num_components_extrap[file_names[fish_num][trace_num]]
            fit_file = sio.loadmat('../extrap/results/'+file_names[fish_num][trace_num]+'.mat')
            lls = fit_file['lls']
            fits = fit_file['fits']
            best_fit_ind = np.argmax(lls[n-1,:])

            # generate extrapolation
            extrap_best = fitting_functions.exponentialModel(trange, fits[n-1,0][best_fit_ind,:])*eye_pos[0]
            sr_data[trace_num] = extrap_best[release_index:] - eye_pos[release_index:]
            sr_data[trace_num] /= sr_data[trace_num][0]

            # generate conservative extrapolation
            conservative_file = sio.loadmat('../extrap/results/'+file_names[fish_num][trace_num]+'_conservative.mat')
            conservative_fit = conservative_file['fit'][0]
            extrap_conservative = fitting_functions.exponentialModel(trange, conservative_fit)*eye_pos[0]
            sr_data_conservative[trace_num] = extrap_conservative[release_index:] - eye_pos[release_index:]
            sr_data_conservative[trace_num] /= sr_data_conservative[trace_num][0]

            sr_tranges[trace_num] = trange[release_index:] - trange[release_index]

        # Fit initial extrapolation
        fits_original, lls_original, sses_original = fitting_functions.fitNEyePositions(sr_tranges, sr_data, max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
        # Save initial extrapolations
        sio.savemat('results/best/'+fish_name+'.mat', {'fits': fits_original, 'sses':sses_original, 'lls':lls_original})

        # Fit initial extrapolation
        fits_conservative, lls_conservative, sses_conservative = fitting_functions.fitNEyePositions(sr_tranges, sr_data_conservative, max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
        # Save initial extrapolations
        sio.savemat('results/conservative/'+fish_name+'_conservative.mat', {'fits': fits_conservative, 'sses':sses_conservative, 'lls':lls_conservative})
