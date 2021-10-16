import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

def fitExponentialWithEndConstrained(data, trange, initial_params, l, end_time, end_val, numComponents):

    # the model
    def model(t, params_):
        returnval = 0
        for i in range(numComponents):
            returnval +=params_[i]*np.exp(-t*params_[numComponents+i])
        return returnval

    # objective function (squared error of model fit)
    def obj_fun(params_):
        return 0.5*sum((model(trange,params_)-data)**2) + 0.5*l*(model(end_time, params_)-end_val)**2

    # Gradient vector
    def jac(params_):
        returnval = np.zeros(len(params_))
        residuals = np.hstack((data - model(trange, params_), l*(end_val - model(end_time, params_))))
        trange_aug = np.hstack((trange, end_time))
        for i in range(numComponents):
            returnval[i] = np.dot(residuals, -np.exp(-trange_aug*params_[numComponents+i]))
        for j in range(numComponents, len(params_)):
            returnval[j] = np.dot(residuals, params_[j-numComponents]*trange_aug*np.exp(-trange_aug*params_[j]))
        return returnval

    opt_result = scipy.optimize.minimize(obj_fun, initial_params, method='TNC', jac=jac, bounds=((0,None),)*len(initial_params), options={'gtol':1e-8, 'ftol':1e-8})
    return opt_result.x, obj_fun(opt_result.x)

def extrapolationAnalysisWithICs(pe_trange, pe_eye_data, pe_fit, delta_time, delta, num_coeffs=0, num_ics = 20, err_dev_lim = 0.1, upper_lim_lmb = 1000, it_count_max=50):
    if num_coeffs == 0:
        num_coeffs = len(pe_fit)//2+1
    best_model = fitting_functions.exponentialModel(delta_time, pe_fit)
    sse_fit = np.sum((pe_eye_data[0]*fitting_functions.exponentialModel(pe_trange, pe_fit) - pe_eye_data)**2)

    best_fit_delta = np.array([])

    lmb_min = 0.
    lmb_max = upper_lim_lmb
    lmb = (lmb_min+lmb_max)/2.
    it_count = 0
    while 1:
        # print("lmb: "+str(lmb))
        fits_const = np.zeros((num_ics, num_coeffs*2))
        sse_const = np.zeros(num_ics)
        sse_ns = np.zeros(num_ics)
        lls_const = np.zeros(num_ics)

        for j in range(num_ics):

            # coeffs = np.random.rand(num_coeffs)
            # coeffs = coeffs/np.sum(coeffs)
            # coeffs.sort()
            coeffs = np.concatenate((pe_fit[:len(pe_fit)//2], np.array([0])))
            ics = np.concatenate((coeffs, pe_fit[-len(pe_fit)//2:], np.array([np.power(10, -1+0.1*np.random.randn())])))
            # ics = coeffs.tolist()+np.power(10.,-(np.arange(-num_coeffs+2,2)+0.1*np.random.randn(num_coeffs))).tolist()
            fits_const[j], sse_const[j] = fitExponentialWithEndConstrained(pe_eye_data, pe_trange, ics, lmb, delta_time, best_model*(1+delta), num_coeffs)
            lls_const[j] = fitting_functions.logLikelihood(np.hstack((pe_eye_data, best_model*(1+delta))), np.hstack((pe_trange, delta_time)), fits_const[j])


        best_fit_index_const = np.argmax(lls_const)
        sse_n = np.sum((pe_eye_data[0]*fitting_functions.exponentialModel(pe_trange, fits_const[best_fit_index_const])-pe_eye_data)**2)
        err_dev = (sse_n - sse_fit)/(sse_fit)
        # print(err_dev)
        if abs(err_dev - err_dev_lim)/err_dev_lim < 1e-1:
            best_fit_delta = fits_const[best_fit_index_const]
            break

        if abs(err_dev) > err_dev_lim:
            lmb_max = lmb
            lmb = (lmb + lmb_min)/2
        else:
            lmb_min = lmb

if __name__ == "__main__":
    traces = ['090711e_0006', '090811c_0002', '090811d_0002', '090811d_0004','091111a_0001',
          '091111a_0003','091111c_0003','091211a_0002','091211a_0005']

    # Load correction factor
    factor_file = sio.loadmat('../../long-fixations/relative_errors.mat')
    delta_time = factor_file['t'][0][0]
    delta = factor_file['delta'][0][0]
    best_num_components = {'090711e_0006':2, '090811c_0002':2, '090811d_0002':2, '090811d_0004':2,'091111a_0001':2,
                            '091111a_0003':3,'091111c_0003':2,'091211a_0002':2,'091211a_0005':2}

    for fish_num in tqdm.trange(len(traces), desc='Trace no.'):
        trange, eye_pos, pe_start_index, displacement_index, release_index, step_pos = fitting_functions.importActiveData('../../../data/active/fixed/'+traces[fish_num]+'.mat')
        fit_file = sio.loadmat('extrap/'+traces[fish_num]+'.mat')
        lls = fit_file['lls']
        fits = fit_file['fits']
        best_fit_ind = np.argmax(lls[best_num_components[traces[fish_num]]-1,:])
        # Fit initial extrapolation
        fit_conservative, lmb = extrapolationAnalysisWithICs(trange[:displacement_index], eye_pos[:displacement_index]/eye_pos[0], fits[best_num_components[traces[fish_num]] - 1,0][best_fit_ind, :], delta_time-0.5, delta)
        # Save initial extrapolations
        sio.savemat('results/'+traces[fish_num]+'_conservative.mat', {'fit': fit_conservative, 'lmb': lmb })
