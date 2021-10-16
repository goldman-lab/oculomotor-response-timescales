import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.io as sio
import scipy.optimize
import tqdm

def loadFitResult(filename):
    fit_file = sio.loadmat(filename)
    sses = fit_file['sses']
    lls = fit_file['lls']
    fits = fit_file['fits']
    return fits, lls, sses

def loadBestFits(filename, n):
    fit_file = sio.loadmat(filename)
    lls = fit_file['lls']
    fits = fit_file['fits']
    best_trace_ind = np.argmax(lls[n-1,:])
    best_fits = np.zeros((fits.shape[1], fits[n-1,0].shape[1]))
    for i in range(fits.shape[1]):
        best_fits[i,:] = fits[n-1,i][best_trace_ind,:]
    return best_fits

short_duration = 10
long_duration = 60
num_ics = 50
if __name__ == "__main__":
    ms222_traces = ['091311a', '091311b', '091311c', '091311d', '091311e', '091311f', '091411a', '091411d', '091411e', '091411f']
    if not os.path.isdir('results/'):
        os.makedirs('results/')
    if not os.path.isdir('results/MS-222/'):
        os.makedirs('results/MS-222/')
    if not os.path.isdir('results/MS-222/distributed'):
        os.makedirs('results/MS-222/distributed')
    for fish_num in tqdm.trange(len(ms222_traces), desc='Fish no.'):
        trange, pe_short, pe_long = fitting_functions.importDataMS222('../../../tools/fixed/MS-222/'+ms222_traces[fish_num]+'.mat')
        best_fits = loadBestFits('../fit/results/MS-222/'+ms222_traces[fish_num]+'.mat', 4)
        timeconstants = best_fits[0, 4:]

        # Bridge the gap between release time and real data
        trange_deconv_mid = np.arange(0, int(0.2304/(72*2e-4))+1)*72*2e-4
        pe_short_deconv_mid = fitting_functions.exponentialModel(trange_deconv_mid, best_fits[0,:])
        pe_long_deconv_mid = fitting_functions.exponentialModel(trange_deconv_mid, best_fits[1,:])

        # Construct 10s hold
        trange_short_deconv_pre = np.arange(0, short_duration, 72*2e-4)
        pe_short_deconv_pre = np.ones(len(trange_short_deconv_pre))
        trange_long_deconv_pre = np.arange(0, long_duration, 72*2e-4)
        pe_long_deconv_pre = np.ones(len(trange_long_deconv_pre))

        trange_short_deconv = np.concatenate((trange_short_deconv_pre, short_duration+trange_deconv_mid, short_duration+0.2304+trange[1:]))
        trange_long_deconv = np.concatenate((trange_long_deconv_pre, long_duration+trange_deconv_mid, long_duration+0.2304+trange[1:]))

        pe_short_deconv = np.concatenate((pe_short_deconv_pre, pe_short_deconv_mid, pe_short[1:]))
        pe_long_deconv = np.concatenate((pe_long_deconv_pre, pe_long_deconv_mid, pe_long[1:]))

        ### For nonlinear fitting, save intermediate results
        if not os.path.isdir('results/MS-222/distributed/'+ms222_traces[fish_num]):
            os.makedirs('results/MS-222/distributed/'+ms222_traces[fish_num])
        cs_nonlin = np.zeros((num_ics, 4))
        costs_nonlin = np.zeros(num_ics)
        for i in tqdm.trange(num_ics, desc='IC no.'):
            ics_ = np.random.rand(4)
            ics_ /= np.sum(ics_)
            f, p, c, cost, grad = fitting_functions.blindDeconvN_NonLin([trange_short_deconv, trange_long_deconv],\
                                                      [pe_short_deconv, pe_long_deconv],\
                                                      [len(trange_short_deconv_pre)+1, len(trange_long_deconv_pre)+1],\
                                                      72*2e-4, np.concatenate((ics_, timeconstants)), method='TNC')
            sio.savemat('results/MS-222/distributed/'+ms222_traces[fish_num]+'/nonlinear'+str(i+1)+'.mat', {'c':c, 'cost':cost, 'grad':grad})
            cs_nonlin[i,:] = c
            costs_nonlin[i] = cost

        ### For linear fitting
        f_linear, p_linear, c_linear, cost_f_linear, cost_p_linear, grad_linear = fitting_functions.blindDeconvN_Linear([trange_short_deconv, trange_long_deconv],\
                                                                                      [pe_short_deconv, pe_long_deconv],\
                                                                                      [len(trange_short_deconv_pre)+1, len(trange_long_deconv_pre)+1],\
                                                                                      72*2e-4, np.concatenate((cs_nonlin[np.argmin(costs_nonlin), :], timeconstants)))
        sio.savemat('results/MS-222/distributed/'+ms222_traces[fish_num]+'.mat', {'c':c_linear, 'cost_f':cost_f_linear, 'cost_p':cost_p_linear, 'grad':grad_linear})
