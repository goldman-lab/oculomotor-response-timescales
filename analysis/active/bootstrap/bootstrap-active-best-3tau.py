import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

import os

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

if __name__ == "__main__":
    # file_names = [('090711e_0006',), ('090811c_0002',), ('090811d_0002','090811d_0004',),
    #    ('091111a_0001', '091111a_0003'), ('091111c_0003',), ('091211a_0002', '091211a_0005')]
    file_names = [('091111a_0001', '091111a_0003'),]
    best_num_components_extrap = {'090711e_0006':2, '090811c_0002':2, '090811d_0002':2, '090811d_0004':2,'091111a_0001':2,
                        '091111a_0003':3,'091111c_0003':2,'091211a_0002':2,'091211a_0005':2}

    num_bootstrap_examples =  100
    num_ics = 50
    if not os.path.isdir('results'):
        os.makedirs('results')
    if not os.path.isdir('results/best'):
        os.makedirs('results/best')

    for fish_num in tqdm.trange(len(file_names), desc='Fish no.'):
        fish_name = file_names[fish_num][0][:-5]
        if not os.path.isdir('results/best/'+fish_name):
            os.makedirs('results/best/'+fish_name)
        if not os.path.isdir('results/best/'+fish_name+'/best_3tau'):
            os.makedirs('results/best/'+fish_name+'/best_3tau')
        n = 3 # best_num_components[fish_name]

        min_len = np.inf
        sr_fits = np.zeros((len(file_names[fish_num]), n*2))
        models = [[] for i in range(len(file_names[fish_num]))]
        tranges = [[] for i in range(len(file_names[fish_num]))]
        samp_vars = np.zeros(len(file_names[fish_num]))

        for trace_num in range(len(file_names[fish_num])):
            trange, data, release_index = prepareStepResponse(file_names[fish_num][trace_num],
                                                num_components_extrap = best_num_components_extrap[file_names[fish_num][trace_num]],
                                                full=False) # NB release_index is chosen as *exclusive* index for slicing in deconvolution procedure!

            # get best fit
            fit_file = sio.loadmat('../fit/results/best/'+fish_name+'.mat')
            lls_fit = fit_file['lls']
            fits = fit_file['fits']
            best_trace_ind = np.argmax(lls_fit[n-1, :])
            sr_fits[trace_num,:] = fits[n-1,0][best_trace_ind,:]
            # calculate the noise variance
            models[trace_num] = fitting_functions.exponentialModel(trange[release_index-1:]-trange[release_index-1], fits[n-1,0][best_trace_ind,:])
            samp_vars[trace_num] = np.sum((data[release_index-1:]/data[release_index-1] - models[trace_num])**2)/float(len((trange[release_index-1:])))
            tranges[trace_num] = trange[release_index-1:]-trange[release_index-1]

        ## Run the bootstap procedure
        bootstrap_fits = np.zeros((num_bootstrap_examples, len(file_names[fish_num]), 2*n))

        for i in tqdm.trange(num_bootstrap_examples,desc='Bootstrap no.',leave=False):
            bootstrap_data = [np.copy(models[i]) for i in range(len(models))]
            for trace_num in range(len(file_names[fish_num])):
                bootstrap_data[trace_num][1:] += np.sqrt(samp_vars[trace_num])*np.random.randn(len(tranges[trace_num])-1)


            fits, lls, sses = fitting_functions.fitNEyePositions(tranges, bootstrap_data, max_num_components=n, min_num_components=n, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
            sio.savemat('results/best/'+fish_name+'/best_3tau/bootstrap_'+str(i+1)+'.mat', {'fits': fits, 'sses':sses, 'lls':lls}, appendmat=False)
            for j in range(len(file_names[fish_num])):
                bootstrap_fits[i,j,:] = fits[n-1,j][np.argmax(lls[n-1,:]), :]

        # save overall results
        sio.savemat('results/best/'+fish_name+'_3tau.mat', {'fits':bootstrap_fits}, appendmat=False)
