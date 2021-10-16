import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.io as sio
import scipy.optimize
import tqdm

def prepareStepResponse(filename, full=True, num_components_extrap = 2, num_components_sr = 3, conservative=False):
    trange, eye_pos, pe_start_index, displacement_index, release_index, step_pos = fitting_functions.importActiveData('../../../data/active/fixed/'+filename+'.mat')

    fish_name = filename[:-5]
    trange_mid = 72*2e-4*(np.arange(1, 16))
    trange_full = np.concatenate((trange[:release_index+1], trange_mid+trange[release_index], trange[release_index+1:]))

    if conservative:
        conservative_file = sio.loadmat('../extrap/results/'+filename+'_conservative.mat')
        conservative_fit = conservative_file['fit'][0]
        if full:
            extrap = fitting_functions.exponentialModel(trange_full, conservative_fit)*eye_pos[0]
            fit_file = sio.loadmat('../fit/results/conservative/'+fish_name+'_conservative.mat')
            lls_fit = fit_file['lls']
            fits = fit_file['fits']
            best_trace_ind = np.argmax(lls_fit[num_components_sr-1, :])
            sr_data_mid = fitting_functions.exponentialModel(trange_mid, fits[num_components_sr-1,0][best_trace_ind,:])*step_pos
        else:
            extrap = fitting_functions.exponentialModel(trange, conservative_fit)*eye_pos[0]

    else:
        extrap_file = sio.loadmat('../extrap/results/'+filename +'.mat')
        lls = extrap_file['lls']
        fits_extrap = extrap_file['fits']
        best_fit_ind = np.argmax(lls[num_components_extrap-1,:])
        if full:
            extrap = fitting_functions.exponentialModel(trange_full, fits_extrap[num_components_extrap-1,0][best_fit_ind,:])*eye_pos[0]
            fit_file = sio.loadmat('../fit/results/best/'+fish_name+'.mat')
            lls_fit = fit_file['lls']
            fits = fit_file['fits']
            best_trace_ind = np.argmax(lls_fit[num_components_sr-1, :])
            sr_data_mid = fitting_functions.exponentialModel(trange_mid, fits[num_components_sr-1,0][best_trace_ind,:])*step_pos
        else:
            extrap = fitting_functions.exponentialModel(trange, fits_extrap[num_components_extrap-1,0][best_fit_ind,:])*eye_pos[0]


    if full:
        eye_data =  eye_pos[displacement_index:]
        sr_data = extrap[displacement_index:] - np.insert(eye_data, release_index-displacement_index+1, sr_data_mid)
        sr_trange = trange_full[displacement_index:] - trange_full[displacement_index]
    else:
        sr_data = extrap[displacement_index:] - eye_pos[displacement_index:]
        sr_trange = trange[displacement_index:] - trange[displacement_index]

    return sr_trange, sr_data, release_index-displacement_index+1

num_ics = 50
if __name__ == "__main__":
    # file_names = [ ('090711e_0006',), ('090811c_0002',), ('090811d_0002','090811d_0004',),
    #      ('091111a_0001', '091111a_0003'), ('091111c_0003',), ('091211a_0002', '091211a_0005')]
    file_names = [ ('091111a_0001', '091111a_0003'), ]
    best_num_components_sr = {'090711e':3, '090811c':3, '090811d':3, '091111a':4, '091111c':3, '091211a':3}
    best_num_components_extrap = {'090711e_0006':2, '090811c_0002':2, '090811d_0002':2, '090811d_0004':2,'091111a_0001':2,
                        '091111a_0003':3,'091111c_0003':2,'091211a_0002':2,'091211a_0005':2}

    if not os.path.isdir('results'):
        os.makedirs('results')
    if not os.path.isdir('results/best'):
        os.makedirs('results/best')
    if not os.path.isdir('results/best/fast'):
        os.makedirs('results/best/fast')

    for fish_num in tqdm.trange(len(file_names), desc='Fish'):
        fish_name = file_names[fish_num][0][:-5]
        n = best_num_components_sr[fish_name]
        ### For nonlinear fitting
        if not os.path.isdir('results/best/fast/'+fish_name):
            os.makedirs('results/best/fast/'+fish_name)

        fit_file = sio.loadmat('../fit/results/best/'+fish_name+'.mat')
        lls_fit = fit_file['lls']
        fits = fit_file['fits']
        best_trace_ind = np.argmax(lls_fit[n-1, :])
        timeconstants = fits[n-1,0][best_trace_ind,n:]
        timeconstants_fast = timeconstants[timeconstants>1]

        tranges = [[] for i in range(len(file_names[fish_num]))]
        eye_pos = [[] for i in range(len(file_names[fish_num]))]
        release_indices = [[] for i in range(len(file_names[fish_num]))]
        for i in range(len(file_names[fish_num])):
            n_extrap = best_num_components_extrap[file_names[fish_num][i]]
            tranges[i], eye_pos[i], release_index = prepareStepResponse(file_names[fish_num][i], num_components_sr=n, num_components_extrap=n_extrap)
            release_indices[i] = release_index
            eye_pos[i] /= eye_pos[i][release_index-1]

        cs_nonlin = np.zeros((num_ics, len(timeconstants_fast)))
        costs_nonlin = np.zeros(num_ics)
        for i in tqdm.trange(num_ics, desc='IC no.'):
            ics_ = np.random.rand(len(timeconstants_fast))
            ics_ /= np.sum(ics_)
            f, p, c, cost, grad = fitting_functions.blindDeconvN_NonLin(tranges, eye_pos, release_indices,\
                                                      72*2e-4, np.concatenate((ics_, timeconstants_fast)), method='TNC')
            sio.savemat('results/best/fast/'+fish_name+'/nonlinear'+str(i+1)+'.mat', {'c':c, 'cost':cost, 'grad':grad})
            cs_nonlin[i,:] = c
            costs_nonlin[i] = cost

        ### For linear fitting
        f_linear, p_linear, c_linear, cost_f_linear, cost_p_linear, grad_linear = fitting_functions.blindDeconvN_Linear(tranges, eye_pos, release_indices,\
                                          72*2e-4, np.concatenate((cs_nonlin[np.argmin(costs_nonlin), :], timeconstants_fast)))
        sio.savemat('results/best/fast/'+fish_name+'.mat', {'c':c_linear, 'cost_f':cost_f_linear, 'cost_p':cost_p_linear, 'grad':grad_linear})
