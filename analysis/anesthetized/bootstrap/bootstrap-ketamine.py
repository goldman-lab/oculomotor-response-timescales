import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

import os

if __name__ == "__main__":
    ketamine_traces = ['63011d','70911i', '70911l', '70911m', '82411p', '82411r']
    num_bootstrap_examples = 100
    num_ics = 50
    if not os.path.isdir('results'):
        os.makedirs('results')
    if not os.path.isdir('results/Ketamine'):
        os.makedirs('results/Ketamine')
    for fish_num in tqdm.trange(len(ketamine_traces), desc='Trace no.'):
        fish_name = ketamine_traces[fish_num]
        if not os.path.isdir('results/Ketamine/'+fish_name):
            os.makedirs('results/Ketamine/'+fish_name)
        trange, pe_short, pe_long = fitting_functions.importDataKetamine('../../../data/anesthetized/fixed/Ketamine/'+ketamine_traces[fish_num]+'.mat')
        ## Load original best fit
        fit_file = sio.loadmat('../fit/results/Ketamine/'+fish_name+'.mat')
        fits = fit_file['fits']
        lls = fit_file['lls']
        best_trace_num = np.argmax(lls[3,:])

        best_fit_short = fits[3,0][best_trace_num]
        best_fit_long = fits[3,1][best_trace_num]

        best_model_short = fitting_functions.exponentialModel(trange, best_fit_short)
        best_model_long = fitting_functions.exponentialModel(trange, best_fit_long)

        samp_var_short = np.sum((pe_short - best_model_short)**2)/float(len(trange))
        samp_var_long = np.sum((pe_long - best_model_long)**2)/float(len(trange))

        bootstrap_fits = np.zeros((num_bootstrap_examples, 2, 2*4)) # 4 component models

        for i in tqdm.trange(num_bootstrap_examples,desc='Bootstrap no.',leave=False):
            ## Generate new random fit from parameters
            bootstap_short = np.copy(best_model_short)
            bootstap_short[1:] += np.sqrt(samp_var_short) * np.random.randn(len(best_model_short)-1)

            bootstrap_long = np.copy(best_model_long)
            bootstrap_long[1:] += np.sqrt(samp_var_long) * np.random.randn(len(best_model_long)-1)

            fits, lls, sses = fitting_functions.fitNEyePositions(trange, np.vstack((bootstap_short, bootstrap_long)), max_num_components=4, min_num_components=4, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
            sio.savemat('results/Ketamine/'+fish_name+'/bootstrap_'+str(i+1)+'.mat', {'fits': fits, 'sses':sses, 'lls':lls}, appendmat=False)
            bootstrap_fits[i,0,:] = fits[3,0][np.argmax(lls[3,:]), :]
            bootstrap_fits[i,1,:] = fits[3,1][np.argmax(lls[3,:]), :]

        # save overall results
        sio.savemat('results/Ketamine/'+fish_name+'.mat', {'fits':bootstrap_fits}, appendmat=False)
