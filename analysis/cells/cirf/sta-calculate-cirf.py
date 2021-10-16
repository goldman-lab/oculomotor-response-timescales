import numpy as np
import scipy.io as sio

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.optimize
import tqdm

num_ics_fit = 100
def correlation(series1, series2):
    series1_mu = np.mean(series1)
    series2_mu = np.mean(series2)
    cov = np.sum((series1 - series1_mu)*(series2-series2_mu))
    return cov/(np.sqrt(np.sum((series1 - series1_mu)**2)*np.sum((series2-series2_mu)**2)))

def calculate_cirfs_and_deconv(filename):
    data_file_original = sio.loadmat('../../../data/cells/fluo/'+filename+'.mat')
    fluo_ipsi_original = data_file_original['FluoIpsi']
    fluo_contra_original = data_file_original['FluoContra']
    eyepos_ipsi = data_file_original['ipsiSTAE6smoo'][0] - data_file_original['null_pos'][0][0]
    eyepos_contra = data_file_original['contraSTAE6smoo'][0] - data_file_original['null_pos'][0][0]

    best_cirf_fits = np.zeros((fluo_contra_original.shape[1], 2))
    best_cirf_r2 = np.zeros(fluo_contra_original.shape[1])
    best_cirf_corr = np.zeros(fluo_contra_original.shape[1])

    fluo_ipsi_means_original = np.mean(fluo_ipsi_original[1537-1000:1537-500,:], axis=0) # 2 s to 1 s before saccade time
    fluo_contra_means_original = np.mean(fluo_contra_original[1537+2000:1537+2500,:], axis=0) # 4 to 5 s after saccade time
    fluo_means_original = np.minimum(fluo_ipsi_means_original, fluo_contra_means_original)

    trange = np.arange(0, 2e-3*4097, 2e-3)

    contra_peak = len(eyepos_contra)-3072 # Traces have been prealigned

    # fit contra fluorescence to single exponential
    for i in tqdm.trange(fluo_contra_original.shape[1], desc='Cell no.', leave=False):
        fluo_data = fluo_contra_original[contra_peak:,i] - fluo_means_original[i]
        cirf_fits_new, cirf_lls_new, cirf_sse_new = fitting_functions.fitNEyePositions(trange[:-contra_peak], np.array([fluo_data]), max_num_components=1, num_ics=num_ics_fit, isConstrained=False)

        mu = np.mean(fluo_data)
        best_cirf_sstot = np.sum((fluo_data - mu)**2)
        best_trace_num = np.argmax(cirf_lls_new[0,:])
        best_cirf_fits[i,:] = cirf_fits_new[0, 0][best_trace_num,:]
        best_cirf_sse = np.sum((fluo_data - fitting_functions.exponentialModel(trange[:-contra_peak], best_cirf_fits[i,:]))**2)

        best_cirf_r2[i] = 1 - best_cirf_sse/best_cirf_sstot # best_cirf_r2s[best_trace_num]

        # Calculate eye pos corr
        cirf_filter = fitting_functions.exponentialModel(trange, best_cirf_fits[i,:])
        cirf_conv_eye_pos = 2e-3*np.convolve(eyepos_contra, cirf_filter)[:len(cirf_filter)]
        fluo_data_saccade = fluo_contra_original[contra_peak-1000:,i]-fluo_means_original[i]

        best_cirf_corr[i] = correlation(fluo_data_saccade, cirf_conv_eye_pos[contra_peak-1000:])

        # print(i, best_cirf_r2[i])
    # save all calculated cirfs
    savedict = {'fits': best_cirf_fits, 'r2': best_cirf_r2, 'corr': best_cirf_corr}
    sio.savemat('results/'+filename+'_contra.mat', savedict, appendmat=False)

if __name__ == "__main__":
    file_names = ('110309FISH1', '110609FISH1',
             '111209FISH2', '111309FISH1', '111509FISH1', '111609FISH4')
    if not os.path.isdir('results'):
        os.makedirs('results')
    for i in tqdm.trange(len(file_names), desc='Fish no.'):
        calculate_cirfs_and_deconv(file_names[i])
