import numpy as np
import scipy.io as sio

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.optimize
import tqdm

num_ics_fit = 100

def exponentialModelConvolved(trange, params, tau_cirf):
    dt = trange[1] - trange[0]
    impulse = np.zeros(len(trange))
    impulse[0] = 1./dt*params[-1];
    exp_model = fitting_functions.exponentialModel(trange, params[:-1])
    cirf = fitting_functions.exponentialModel(trange,np.array([1, tau_cirf]))
    return dt*np.convolve(exp_model+impulse, cirf)[:len(exp_model)]

def logLikelihoodConvolved(data, trange, params, tau_cirf):
    model = exponentialModelConvolved(trange, params, tau_cirf)
    samp_var = 1./len(trange)*np.sum((model-data)**2)
    return -len(trange)/2.*(np.log(2*np.pi*samp_var)+1)

def fitConvolvedExponential(data, trange, tau_cirf, initial_params, numComponents, isConstrained=False, min_bound=0, max_bound=None, method='L-BFGS-B', gtol=1e-8, ftol=1e-8, maxiter=1000):

    dt = trange[1] - trange[0]
    impulse = np.zeros(len(trange))
    impulse[0] = 1./dt;
    cirf = fitting_functions.exponentialModel(trange, np.array([1, tau_cirf]))

    # if exponential is constrained, divide parameters
    def subparams(params_):
        returnparams = np.copy(params_)
        if isConstrained:
            returnparams[:numComponents] /= np.sum(returnparams[:numComponents])
        return returnparams

    # the model
    def model(t, params_):
        exp_model = fitting_functions.exponentialModel(t, subparams(params_[:-1]))
        return dt*np.convolve(exp_model + params_[-1]*impulse, cirf)[:len(exp_model)]

    # objective function (squared error of model fit)
    def obj_fun(params_):
        return 0.5*np.sum((data - model(trange, params_))**2)

    # Gradient vector
    def jac(params_):
        returnval = np.zeros(len(params_))
        coeff_sum = sum(params_[:numComponents])
        model_val = model(trange, params_)
        residuals = data - model_val
        for i in range(numComponents):
            exp = np.exp(-trange*params_[numComponents+i])
            if isConstrained:
                drdc = (dt*np.convolve(exp, cirf)[:len(exp)] - model_val)/coeff_sum
                drdb = -params_[i]/coeff_sum*dt*np.convolve(trange*exp, cirf)[:len(exp)]
            else:
                drdc = dt*np.convolve(exp, cirf)[:len(exp)]
                drdb = -params_[i]*dt*np.convolve(trange*exp, cirf)[:len(exp)]
            returnval[i] = np.dot(residuals, -drdc)
            returnval[numComponents + i] = np.dot(residuals, -drdb)
        returnval[-1] = np.dot(residuals, -cirf)
        return returnval

    bounds = ((0, None),)*numComponents + ((min_bound, max_bound),)*numComponents + ((0,None),)
    opt_result = scipy.optimize.minimize(obj_fun, initial_params, jac=jac, bounds=bounds, method = method,\
                    options={'gtol': gtol, 'ftol':ftol, 'maxiter':maxiter})
    if isConstrained:
        opt_result.x[:numComponents] /= np.sum(opt_result.x[:numComponents])
    return opt_result.x, obj_fun(opt_result.x), opt_result.success

def fitConvolvedEyePosition(trange, data, tau_cirf, num_ics=20, isConstrained = False, max_num_components = 6, min_bound = 0, max_bound=None):
    def getCoefficients(initial_tau):
        numComponents = len(initial_tau)
        expmatrix = np.zeros((len(trange), numComponents))
        for i in range(numComponents):
            expmatrix[:,i] = np.exp(-trange*initial_tau[i])
        optresult = scipy.optimize.lsq_linear(expmatrix, data, bounds=(0, np.inf), method='bvls')
        return optresult.x

    sse = np.zeros((max_num_components,num_ics))
    lls = np.zeros((max_num_components,num_ics))
    fits = np.array(np.zeros((max_num_components, )), dtype=object)

    dt = trange[1] - trange[0]
    for i in tqdm.trange(1, max_num_components+1, desc="Component no.", leave=False):
        # Calculate fits from random starting conditions
        # We pick three random initial coefficients, and sort them so that the first coefficient is the smallest
        # Then, for the time constants, we pick 10^(-i+2 to 2), with Gaussian noise (std=0.1) added to the exponents
        # (The fit uses beta = 1/tau, so we take the negative exponents)
        fits[i-1] = np.empty((num_ics, 2*i+1))*np.nan
        #if notebookMode:
        #    innerrange = tnrange(num_ics, desc='Initial condition no.:', leave=False)
        #else:
        for j in tqdm.trange(num_ics, desc="IC no.",leave=False):
            for k in range(fitting_functions.ITER_LIM_PER_IC):
                taus = np.power(10.,-(np.linspace(-1,1,i)+0.1*np.random.randn(i)))
                coeffs = getCoefficients(taus) # would assume an instantaneous CIRF
                if isConstrained:
                    coeffs = coeffs / np.sum(coeffs)
                ics = np.concatenate((coeffs, taus, np.array([1])))
                fit_temp, sse_temp, succ = fitConvolvedExponential(data, trange, tau_cirf, ics, i, isConstrained=isConstrained, min_bound=min_bound, max_bound=max_bound, method='TNC')
                if(succ):
                    break
            fits[i-1][j] = fit_temp
            sse[i-1,j] = sse_temp
            lls[i-1,j] = logLikelihoodConvolved(data, trange, fits[i-1][j], tau_cirf)
        # Pick the best fit
    return fits, lls, 2*sse

def fit_responses(filename):
    data_file = sio.loadmat('../../../data/cells/fluo/'+filename+'.mat')
    fluo_ipsi = data_file['FluoIpsi']
    fluo_contra = data_file['FluoContra']
    eyepos_ipsi = data_file['ipsiSTAE6'][0] - data_file['null_pos'][0][0]

    fluo_ipsi_means = np.mean(fluo_ipsi[1537-1000:1537-500,:], axis=0)
    fluo_contra_means= np.mean(fluo_contra[1537+2000:1537+2500,:], axis=0)
    fluo_means = np.minimum(fluo_ipsi_means, fluo_contra_means)

    trange = np.arange(0, 2e-3*4097, 2e-3)

    ipsi_peak = np.argmax(eyepos_ipsi)

    fits_file = sio.loadmat('../cirf/results/'+filename+'_contra.mat')
    best_cirf_fits = fits_file['fits']
    best_cirf_r2s = fits_file['r2'][0]
    best_cirf_corr = fits_file['corr'][0]

    good_cells = (best_cirf_r2s >= 0.5)*(best_cirf_corr > 0.5)

    if not os.path.isdir('results/'+filename):
        os.makedirs('results/'+filename)

    best_ipsi_fits = np.array(np.zeros(len(good_cells)), dtype=object)
    k_cirfs = np.zeros(len(good_cells))
    indices = np.arange(len(best_cirf_r2s))[good_cells]

    j = 0
    for i in tqdm.trange(fluo_ipsi.shape[1], desc='Cell no.', leave=False):
        if best_cirf_r2s[i] < 0.5 or best_cirf_corr[i] <= 0.5:
            continue
        fr_fits_new, fr_lls_new, fr_sse_new = fitConvolvedEyePosition(trange[:-ipsi_peak], fluo_ipsi[ipsi_peak:,i]-fluo_means[i], best_cirf_fits[i,-1], max_num_components=3, num_ics=num_ics_fit)
        # print(fr_aics_new.shape)
        # save each cell separately so we can see progress
        sio.savemat('results/'+filename+'/fluo_delayed_saccade/cell_'+str(j+1)+'.mat', {'fits':fr_fits_new, 'lls':fr_lls_new, 'sses':fr_sse_new}, appendmat=False)

        best_traces = np.argmax(fr_lls_new, axis=1)
        sses = fr_sse_new[[0,1,2], best_traces]
        pct_change_sse = (sses[1:]-sses[:-1])/sses[:-1]

        ## Choose number of components as where % change SSE to the next number is less than 10%
        if np.abs(pct_change_sse[0]) < 1e-2:
            best_num_components = 1
        elif np.abs(pct_change_sse[1]) < 1e-2:
            best_num_components = 2
        else:
            best_num_components = 3

        best_ipsi_fits[j] = fr_fits_new[best_num_components-1][best_traces[best_num_components-1], :]
        k_cirfs[j] = best_cirf_fits[i,-1]
        j += 1

    sio.savemat('results/'+filename+'_ipsi.mat', {'fits':best_ipsi_fits, 'k_cirf':k_cirfs, 'indices':indices})

if __name__ == "__main__":
    file_names = ('110309FISH1', '110609FISH1',
             '111209FISH2', '111309FISH1', '111509FISH1', '111609FISH4')
    for fish_num in tqdm.trange(len(file_names), desc='Fish no.'):
        fit_responses(file_names[fish_num])
