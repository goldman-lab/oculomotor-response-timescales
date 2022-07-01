import numpy as np
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
import scipy.io as sio
import tqdm
import tqdm.notebook

version = '20220308'
diagnosticMode = False # set to True to print outputs to console
ITER_LIM_PER_IC = 20 # Number of restarts to use per starting point in exponential fits if fitting fails


###### Functions to handle data files

def importActiveData(filename, fixed=True):
    """Load data for an active state displacement experiment .mat file"""
    data_file = sio.loadmat(filename)

    # Eye position data
    full_data = data_file['data']
    if not fixed:
        full_data = np.reshape(full_data, (full_data.shape[1],))
    else:
        full_data = full_data[0]

    # Time range
    full_trange = data_file['trange']
    if not fixed:
        full_trange = np.reshape(full_trange, (full_trange.shape[1],))
    else:
        full_trange = full_trange[0]

    if not fixed:
        pe_start_index = data_file['pe_start_index'][0][0] # Time index of saccade
    else:
        pe_start_index = 0
    displacement_index = data_file['displacement_index'][0][0] # Time index of displacement
    release_index = data_file['release_index'][0][0] # Time index of release from displacement

    if not fixed:
        step_pos = data_file['step_pos'][0][0] # Eye position during the displacement
    else:
        step_pos = full_data[release_index]

    return full_trange, full_data, pe_start_index, displacement_index, release_index, step_pos

def importDataKetamine(filename):
    """Load data for a ketamine-anesthetized displacement experiment .mat file."""
    data_file = sio.loadmat(filename)
    eye_pos_data_short = data_file['eye_pos_15s'][0]
    eye_pos_data_long = data_file['eye_pos_90s'][0]
    trange = np.concatenate((np.array([0]), 0.2304+np.arange(0,len(eye_pos_data_short)-1)/(5000./72)))
    return trange, eye_pos_data_short, eye_pos_data_long

def importDataMS222(filename):
    """Load data for an MS-222-anesthetized displacement experiment .mat file."""
    data_file = sio.loadmat(filename)
    eye_pos_data_short = data_file['eye_pos_10s'][0]
    eye_pos_data_long = data_file['eye_pos_60s'][0]
    trange = np.concatenate((np.array([0]), 0.2304+np.arange(0,len(eye_pos_data_short)-1)/(5000./72)))
    return trange, eye_pos_data_short, eye_pos_data_long

###### Functions to perform fits, and related

def exponentialModel(trange, params):
    """Generate time course of an n-component multiexponential model given a set of parameters.

    Arguments:
        trange:
            array of time points to evaluate the model
        params:
            array of parameters of length 2n; the first n terms are the
            coefficients for each of the components, the second n terms are the
            inverse time constants for each component.

    Returns the time course of the model, evaluated at each point in trange.
    """
    numComponents = len(params)//2
    model = np.sum(np.vstack([params[i]*np.exp(-trange*params[numComponents+i])\
            for i in range(numComponents)]), axis=0)
    return model

def logLikelihood(data, trange, params):
    """Returns the log-likelihood of an n-component multiexponential model given a set of parameters.

    Arguments:
        data, trange:
            eye position and corresponding time value
        params:
            array of parameters of length 2n; the first n terms are the
            coefficients for each of the components, the second n terms are the
            inverse time constants for each component.
    """
    samp_var = 1./len(trange)*np.sum((exponentialModel(trange, params)-data)**2)
    return -len(trange)/2.*(np.log(2*np.pi*samp_var)+1)

def fitNExponentials_LS(data_, trange_, initial_params, isConstrained = False, coeff_min = 0, coeff_max = None, inverse_tau_min = 0, inverse_tau_max=None, method='L-BFGS-B', gtol=1e-8, ftol=1e-8, maxiter=1000):
    """Simultaneously fit M n-component multiexponential models using a squared error cost function

    Arguments:
        data:
            a numpy array of size M x T, or
            a list of length M containing the eye position data to be fit
        trange:
            a numpy array of size T, or
            a list of length M containing the time points corresponding to the data
        initial_params:
            array of length (M+1)*n; initial parameter guess to provide to the
            fitting algorithm, where the last n entries are the inverse time constants
            of the model, and the first M*n entries are coefficients corresponding
            to each individual fit.
        isConstrained:
            True if sum of coefficients for each model should equal 1
        inverse_tau_min:
            Lower bound for inverse time constant fits
        inverse_tau_max:
            Upper bound for inverse time constant fits (None = infinity)
        method:
            Optimization algorithm type, either 'L-BFGS-B' or 'TNC'
        gtol:
            Gradient tolerance parameter for fitting algorithm
        ftol:
            Cost function tolerance parameter for fitting algorithm

    Returns model fit parameters, sum of squared errors, and whether the
        algorithm terminated successfully.
    """
    if isinstance(data_, np.ndarray):
        numModels = data_.shape[0]
        data = [data_[j,:] for j in range(numModels)]
        trange = [trange_,]*numModels
    else:
        numModels = len(data_)
        data = data_
        trange = trange_
    numComponents = len(initial_params)//(numModels+1)
    weighting = np.array([1/len(trange[j]) for j in range(numModels)], dtype='float')
    # weighting /= np.sum(weighting)

    def subparams(params_, model_num):
        """Get only the parameters corresponding to model_num"""
        timeconstants = np.copy(params_[-numComponents:])
        coeffs = np.copy(params_[ model_num*numComponents:((model_num+1)*numComponents)])
        if isConstrained:
            coeffs /= np.sum(coeffs)
        returnparams = np.concatenate((coeffs, timeconstants))
        return returnparams

    def model(t, params_, model_num):
        return exponentialModel(t, subparams(params_, model_num))

    def obj_fun(params_):
        """Get the value of the objective function given parameters params_"""
        sqerr = 0
        for n in range(numModels):
            sqerr += weighting[n]*np.sum((data[n]-model(trange[n], params_, n))**2)
        return 0.5*sqerr

    def jac(params_):
        """Calculate the gradient given parameters params_"""
        returnval = np.zeros(len(params_))
        coeff_sum = np.zeros(numModels)
        model_vals = [[],]*numModels # np.zeros(data.shape)
        residuals = [[],]*numModels # np.zeros(data.shape)
        for n in range(numModels):
            coeff_sum[n] = np.sum(params_[n*numComponents:(n+1)*numComponents])
            model_vals[n] = model(trange[n], params_, n)
            residuals[n] = weighting[n]*(data[n]-model_vals[n])
        for i in range(numComponents):
            for n in range(numModels):
                exp_ = np.exp(-trange[n]*params_[-numComponents+i])
                if isConstrained:
                    drdc = (exp_ - model_vals[n])/coeff_sum[n]
                    drdb = -params_[n*numComponents+i]*trange[n]/coeff_sum[n]*exp_
                else:
                    drdc = exp_
                    drdb = -params_[n*numComponents+i]*trange[n]*exp_
                returnval[n*numComponents+i] = np.dot(residuals[n], -drdc)
                returnval[-numComponents+i] += np.dot(residuals[n], -drdb)
        return returnval

    # Optimization bounds
    bounds = ((coeff_min, coeff_max),)*(numModels*numComponents) + ((inverse_tau_min, inverse_tau_max),)*numComponents

    opt_result = scipy.optimize.minimize(obj_fun, initial_params, jac=jac, bounds=bounds, method = method,\
                    options={'disp':False, 'gtol': gtol, 'ftol':ftol, 'maxiter':maxiter})
    if isConstrained:
        for n in range(numModels):
            opt_result.x[n*numComponents:(n+1)*numComponents] /= np.sum(opt_result.x[n*numComponents:(n+1)*numComponents])
    if not opt_result.success and diagnosticMode:
        print(opt_result.message)
    return opt_result.x, obj_fun(opt_result.x), opt_result.success

def fitNExponentials_Likelihood(data_, trange_, initial_params, isConstrained=False, coeff_min = 0, coeff_max = None, inverse_tau_min = 0, inverse_tau_max=None, method='L-BFGS-B', gtol=1e-8, ftol=1e-8, maxiter=1000):
    """Simultaneously fit M n-component multiexponential models using a log-likelihood cost function

    Arguments:
        data:
            a numpy array of size M x T, or
            a list of length M containing the eye position data to be fit
        trange:
            a numpy array of size T, or
            a list of length M containing the time points corresponding to the data
        initial_params:
            array of length (M+1)*n; initial parameter guess to provide to the
            fitting algorithm, where the last M entries are the inverse variances
            for each model, the prior n entries are the inverse time constants,
            and the first M*n entries are coefficients corresponding to each individual fit.
        isConstrained:
            True if sum of coefficients for each model should equal 1
        inverse_tau_min:
            Lower bound for inverse time constant fits
        inverse_tau_max:
            Upper bound for inverse time constant fits
        method:
            Optimization algorithm type, either 'L-BFGS-B' or 'TNC'
        gtol:
            Gradient tolerance parameter for fitting algorithm
        ftol:
            Cost function tolerance parameter for fitting algorithm

    Returns model fit parameters, sum of squared errors, and whether the
        algorithm terminated successfully.
    """
    if isinstance(data_, np.ndarray):
        numModels = data_.shape[0]
        data = [data_[j,:] for j in range(numModels)]
        trange = [trange_,]*numModels
    else:
        numModels = len(data_)
        data = data_
        trange = trange_
    numComponents = len(initial_params)//(numModels+1)

    def subparams(params_, model_num):
        """Get only the parameters corresponding to model_num"""
        timeconstants = np.copy(params_[numModels*numComponents:-numModels])
        coeffs = np.copy(params_[ model_num*numComponents:((model_num+1)*numComponents)])
        if isConstrained:
            coeffs /= np.sum(coeffs)
        returnparams = np.concatenate((coeffs, timeconstants))
        return returnparams

    def model(t, params_, model_num):
        return exponentialModel(t, subparams(params_, model_num))

    def obj_fun(params_):
        """Get the value of the objective function given parameters params_"""
        l = 0
        for n in range(numModels):
            l += len(trange[n])/2*np.log(2*np.pi/params_[-numModels+n])+1./2*params_[-numModels+n]*np.sum((data[n]-model(trange[n], params_,n))**2)
        return l

    def jac(params_):
        returnval = np.zeros(len(params_))
        coeff_sum = np.zeros(numModels)
        model_vals = [[],]*numModels # np.zeros(data.shape)
        residuals = [[],]*numModels # np.zeros(data.shape)
        for n in range(numModels):
            coeff_sum[n] = np.sum(params_[n*numComponents:(n+1)*numComponents])
            model_vals[n] = model(trange[n], params_, n)
            residuals[n] = data[n]-model_vals[n]
        for i in range(numComponents):
            for n in range(numModels):
                exp_ = np.exp(-trange[n]*params_[numModels*numComponents+i])
                if isConstrained:
                    drdc = (exp_ - model_vals[n])/coeff_sum[n]
                    drdb = -params_[n*numComponents+i]/coeff_sum[n]*trange[n]*exp_
                else:
                    drdc = exp_
                    drdb = -params_[n*numComponents+i]*trange[n]*exp_
                returnval[n*numComponents+i] = -params_[-numModels+n]*np.dot(residuals[n], drdc)
                returnval[numModels*numComponents+i] += -params_[-numModels+n]*np.dot(residuals[n], drdb)
        for n in range(numModels):
            returnval[-numModels+n] = -len(trange[n])/(2*params_[-numModels+n]) + 0.5*np.sum(residuals[n]**2)
        return returnval

    # Bounds
    bounds = ((coeff_min, coeff_max),)*(numModels*numComponents) + ((inverse_tau_min, inverse_tau_max),)*numComponents + ((0,None),)*numModels

    opt_result = scipy.optimize.minimize(obj_fun, initial_params, jac=jac, bounds=bounds, method = method,\
                    options={'disp':False, 'gtol': gtol, 'ftol':ftol, 'maxiter':maxiter})
    if isConstrained:
        for n in range(numModels):
            opt_result.x[n*numComponents:(n+1)*numComponents] /= np.sum(opt_result.x[n*numComponents:(n+1)*numComponents])
    if not opt_result.success and diagnosticMode:
        print(opt_result.message)
    return opt_result.x, -obj_fun(opt_result.x), opt_result.success


def fitNEyePositions(tranges, data, use_likelihood = True, min_num_components=1, max_num_components = 6, num_ics=20, isConstrained = True, inverse_tau_min = 0, inverse_tau_max=None, verbose=False, notebookMode=False):
        """Fit M n-component multiexponential models to data

        Arguments:
            trange:
                a numpy array of size T, or
                a list of length M containing the time points corresponding to the data
            data:
                a numpy array of size M x T, or
                a list of length M containing the eye position data to be fit
            use_likelihood:
                True to first use squared error cost function, then log likelihood cost function
                False to use only squared error cost function
            min_num_components, max_num_components:
                Function will run optimization algorithm for n = min_num_components to max_num_components
            num_ics:
                Number of random initial conditions from which to start optimization
            isConstrained:
                True if sum of coefficients for each model should equal 1
            inverse_tau_min:
                Lower bound for inverse time constant fits
            inverse_tau_max:
                Upper bound for inverse time constant fits
            verbose:
                True to print diagnostic information to console
            notebookMode:
                True to use tqdm notebook widget

        Returns array of all model fit parameters, sum of squared errors, and log likelihoods.
        If min_num_components > 1, array will be left empty for indices < min_num_components
        """
        try:
            num_models = data.shape[0]
            return fitNEyePositionsArray(tranges, data, use_likelihood=use_likelihood, min_num_components=min_num_components, max_num_components=max_num_components, num_ics=num_ics, isConstrained=isConstrained, inverse_tau_max=inverse_tau_max, inverse_tau_min=inverse_tau_min, notebookMode=notebookMode)
        except:
            return fitNEyePositionsList(tranges, data, use_likelihood=use_likelihood, min_num_components=min_num_components, max_num_components=max_num_components, num_ics=num_ics, isConstrained=isConstrained, inverse_tau_max=inverse_tau_max, inverse_tau_min=inverse_tau_min, notebookMode=notebookMode)

def fitNEyePositionsArray(trange, data, use_likelihood = True, min_num_components = 1, max_num_components = 6, num_ics=20, isConstrained = True, inverse_tau_min = 0, inverse_tau_max=None, verbose=False, notebookMode = False):
    def getCoefficients(initial_tau, n):
        numComponents = len(initial_tau)
        expmatrix = np.zeros((len(trange), numComponents))
        for i in range(numComponents):
            expmatrix[:,i] = np.exp(-trange*initial_tau[i])
        optresult = scipy.optimize.lsq_linear(expmatrix, data[n,:], bounds=(0, np.inf), method='bvls')
        return optresult.x

    sse = np.empty((max_num_components,num_ics))*np.nan
    lls = np.empty((max_num_components,num_ics))*np.nan

    number_of_models = data.shape[0]
    fits = np.array(np.zeros((max_num_components, number_of_models)), dtype=object)

    if notebookMode:
        counter = tqdm.notebook.trange
    else:
        counter = tqdm.trange

    for i in counter(min_num_components, max_num_components+1, desc='Component', leave=False):
        for n in range(number_of_models):
            # initialize fit
            fits[i-1,n] = np.empty((num_ics, 2*i))*np.nan
        for j in counter(num_ics, desc='IC', leave=False):
            for k in range(ITER_LIM_PER_IC):
                taus = np.power(10.,-(np.linspace(-1,2,i)+np.array([0.1,]+[0.1,]*(i-1))*np.random.randn(i)))
                coeffs = np.zeros(i*number_of_models)
                for n in range(number_of_models):
                    coeff_temp = getCoefficients(taus, n)
                    coeffs[n*i:(n+1)*i] = coeff_temp / np.sum(coeff_temp)
                ics = np.concatenate((coeffs, taus))
                fits_temp,sse_temp, succ = fitNExponentials_LS(data, trange, ics, isConstrained=isConstrained, inverse_tau_min=inverse_tau_min, inverse_tau_max=inverse_tau_max, method='TNC', ftol=1e-8, gtol=1e-8, maxiter=10000)
                if(succ):
                    trial_fits = np.zeros((number_of_models, 2*i))
                    lls[i-1, j] = 0
                    for n in range(number_of_models):
                        fits[i-1,n][j,:] = np.concatenate((fits_temp[n*i:(n+1)*i], fits_temp[-i:]))
                        trial_fits[n,:] = fits[i-1,n][j,:]
                        lls[i-1, j] += logLikelihood(data[n,:], trange, trial_fits[n,:])
                    sse[i-1,j] = sse_temp
                    break
                if verbose:
                    tqdm.tqdm.write('Retrying n = %d, trial = %d' % (i, j))
            if use_likelihood and succ:
                samp_var = np.zeros(number_of_models)
                for n in range(number_of_models):
                    samp_var[n] = float(len(trange))/np.sum((data[n,:] - exponentialModel(trange, trial_fits[n,:]))**2)
                fits_temp_likelihood, likelihoods, succ_l = fitNExponentials_Likelihood(data, trange, np.concatenate((fits_temp, samp_var)), isConstrained=isConstrained, inverse_tau_min=inverse_tau_min, inverse_tau_max=inverse_tau_max, method='TNC', ftol=1e-15, gtol=1e-15, maxiter=10000)
                if(succ_l):
                    lls[i-1,j] = likelihoods
                    sse[i-1,j] = 0
                    timeconstants = fits_temp_likelihood[number_of_models*i:-number_of_models]
                    for n in range(number_of_models):
                        fits[i-1,n][j,:] = np.concatenate((fits_temp_likelihood[n*i:(n+1)*i], timeconstants))
                        sse[i-1,j] += 0.5*np.sum((data[n,:] - exponentialModel(trange,fits[i-1,n][j,:]))**2)
    # Pick the best fit
    # sse has factor of 0.5 from cost function
    return fits, lls, 2*sse #, full_fits

def fitNEyePositionsList(tranges, data, use_likelihood = True, min_num_components=1, max_num_components = 6, num_ics=20, isConstrained = True, inverse_tau_min = 0, inverse_tau_max=None, verbose=False, notebookMode = False):
    def getCoefficients(initial_tau, n):
        numComponents = len(initial_tau)
        expmatrix = np.zeros((len(tranges[n]), numComponents))
        for i in range(numComponents):
            expmatrix[:,i] = np.exp(-tranges[n]*initial_tau[i])
        optresult = scipy.optimize.lsq_linear(expmatrix, data[n], bounds=(0, np.inf), method='bvls')
        return optresult.x

    sse = np.empty((max_num_components,num_ics))*np.nan
    lls = np.empty((max_num_components,num_ics))*np.nan

    number_of_models = len(data) # data.shape[0]
    fits = np.array(np.zeros((max_num_components, number_of_models)), dtype=object)

    if notebookMode:
        counter = tqdm.notebook.trange
    else:
        counter = tqdm.trange

    for i in counter(min_num_components, max_num_components+1, desc='Component', leave=False):
        for n in range(number_of_models):
            # initialize fit
            fits[i-1,n] = np.empty((num_ics, 2*i))*np.nan
        for j in counter(num_ics, desc='IC', leave=False):
            for k in range(ITER_LIM_PER_IC):
                taus = np.power(10.,-(np.linspace(-1,2,i)+np.array([0.1,]+[0.1,]*(i-1))*np.random.randn(i)))
                coeffs = np.zeros(i*number_of_models)
                for n in range(number_of_models):
                    coeff_temp = getCoefficients(taus, n)
                    coeffs[n*i:(n+1)*i] = coeff_temp / np.sum(coeff_temp)
                ics = np.concatenate((coeffs, taus))
                fits_temp,sse_temp, succ = fitNExponentials_LS(data, tranges, ics, isConstrained=isConstrained, inverse_tau_min=inverse_tau_min, inverse_tau_max=inverse_tau_max, method='TNC', ftol=1e-8, gtol=1e-8, maxiter=10000)
                if(succ):
                    trial_fits = np.zeros((number_of_models, 2*i))
                    lls[i-1, j] = 0
                    for n in range(number_of_models):
                        fits[i-1,n][j,:] = np.concatenate((fits_temp[n*i:(n+1)*i], fits_temp[-i:]))
                        trial_fits[n,:] = fits[i-1,n][j,:]
                        lls[i-1, j] += logLikelihood(data[n], tranges[n], trial_fits[n,:])
                    sse[i-1,j] = sse_temp
                    break
                if verbose:
                    tqdm.tqdm.write('Retrying n = %d, trial = %d' % (i, j))
            if use_likelihood and succ:
                samp_var = np.zeros(number_of_models)
                for n in range(number_of_models):
                    samp_var[n] = float(len(tranges[n]))/np.sum((data[n] - exponentialModel(tranges[n], trial_fits[n,:]))**2)
                fits_temp_likelihood, likelihoods, succ_l = fitNExponentials_Likelihood(data, tranges, np.concatenate((fits_temp, samp_var)), isConstrained=isConstrained, inverse_tau_min=inverse_tau_min, inverse_tau_max=inverse_tau_max, method='TNC', ftol=1e-15, gtol=1e-15, maxiter=10000)
                if(succ_l):
                    lls[i-1,j] = likelihoods
                    sse[i-1,j] = 0
                    timeconstants = fits_temp_likelihood[number_of_models*i:-number_of_models]
                    for n in range(number_of_models):
                        fits[i-1,n][j,:] = np.concatenate((fits_temp_likelihood[n*i:(n+1)*i], timeconstants))
                        sse[i-1,j] += np.sum((data[n] - exponentialModel(tranges[n],fits[i-1,n][j,:]))**2)
                else:
                    tqdm.tqdm.write('Likelihood fit failed.')
    return fits, lls, sse

def AIC(data, trange, params):
    """Returns the AIC of a model fit to the data."""
    numModels = data.shape[0]
    numComponents = params.shape[1]//2
    aic = 0
    for n in range(numModels):
        aic -= 2*logLikelihood(data[n,:], trange, params[n,:])
    aic += 2*((numModels+1)*numComponents+numModels)
    return aic

def BIC(data, trange, params):
    """Returns the BIC of a model fit to the data."""
    numModels = data.shape[0]
    numComponents = params.shape[1]//2
    bic = 0
    for n in range(numModels):
        bic -= 2*logLikelihood(data[n,:], trange, params[n,:])
    bic += np.log(len(trange)*numModels)*((numModels+1)*numComponents+numModels)
    return bic

###### Blind deconvolution methods

def blindDeconvN_NonLin(tranges, data, release_indices, dt, ics, mu=1, scale_factor = 1, verbose=False, method='L-BFGS-B', ftol=None, gtol=None):
    """Performs blind deconvolution of M eye position traces with a n-component multiexponential plant with unknown coefficients using nonlinear optimization,
    assuming applied force becomes zero at some point.

    Arguments:
        tranges, data:
            lists of M arrays containing time points and eye position, respectively
        release_indices:
            list containing index of time array after which applied force is zero, for each of the M inputs
        dt:
            time step size
        ics:
            array of n initial guesses for the plant coefficients, followed by corresponding known inverse plant time constants
        mu:
            scale of weighting value to compensate for M input traces having different lengths
        scale_factor:
            scaling applied to cost function
        verbose:
            True to print diagnostic information to the console
        method, ftol, gtol:
            optimization algorithm and tolerances to be used (passed directly to scipy.optimize.minimize)

    Returns:
        list of M applied force traces,
        list of M plant time courses (length same as tranges[m]),
        coefficients of plant,
        cost function at end of optimization, and
        gradient of cost function at end of optimization.
    """
    numComponents = len(ics)//2
    numModels = len(tranges)
    timeconstants = ics[-numComponents:]

    mu_vecs = [0,]*numModels
    dd = np.array([])
    for n in range(numModels):
        mu_vec = np.ones(len(tranges[n]))*np.sqrt(scale_factor/(len(tranges[n])-release_indices[n]))
        mu_vec[:release_indices[n]] = np.sqrt(mu*scale_factor/(release_indices[n]))
        mu_vecs[n] = mu_vec
        dd = np.concatenate((dd, mu_vec*data[n]))

    def pmatrix(x, p, num):
        mu_vec = mu_vecs[num]
        conv_p = dt*np.convolve(p, x, mode='full')
        return mu_vec*conv_p[:len(p)]
    def pmatrixT(x,p, num):
        mu_vec = mu_vecs[num]
        release_index = release_indices[num]
        corr_p = dt*np.correlate(mu_vec*x, p, mode='full')
        return corr_p[-len(x):-len(x)+release_index]

    def fmatrix(x, f):
        returnvec = np.array([])
        for n in range(numModels):
            e_ = exponentialModel(tranges[n], np.concatenate((x, timeconstants)))
            conv_f_ = dt*np.convolve(f[n], e_, mode='full')
            returnvec = np.concatenate((returnvec, mu_vecs[n]*conv_f_[:len(f[n])]))
        return returnvec
    def fmatrixT(x,f):
        outvec = np.zeros(numComponents)
        mu_vec = np.concatenate(mu_vecs)
        xx = mu_vec*x
        for i in range(numComponents):
            tempvec = np.array([])
            for n in range(numModels):
                tempvec_ = dt*np.convolve(f[n],np.exp(-tranges[n]*timeconstants[i]), mode='full')
                tempvec = np.concatenate((tempvec, tempvec_[:len(f[n])]))
            outvec[i] = np.dot(tempvec, xx)
        return outvec

    def obj_fun(x):
        fs = [0,]*numModels
        for n in range(numModels):
            f_ = np.zeros(len(tranges[n]))
            if n == 0:
                f_[:release_indices[0]] = x[:release_indices[0]]
            else:
                f_[:release_indices[n]] = x[int(np.sum(release_indices[:n])):int(np.sum(release_indices[:n]))+release_indices[n]]
            fs[n] = f_
        model = fmatrix(x[-numComponents:], fs)
        return np.sum((model-dd)**2)

    def jac(x):
        grad = np.zeros(sum(release_indices) + numComponents)
        fs = [0,]*numModels
        ps = [0,]*numModels
        models = [0,]*numModels
        for n in range(numModels):
            f_ = np.zeros(len(tranges[n]))
            if n == 0:
                f_[:release_indices[n]] = x[:release_indices[n]]
            else:
                f_[:release_indices[n]] = x[int(np.sum(release_indices[:n])):int(np.sum(release_indices[:n]))+release_indices[n]]
            fs[n] = f_
            p_ = exponentialModel(tranges[n], np.concatenate((x[-numComponents:], timeconstants)))

            model_ = pmatrix(f_, p_, n)
            grad_ = pmatrixT(model_ - mu_vecs[n]*data[n], p_, n)
            if n == 0:
                grad[:release_indices[0]] = grad_
            else:
                grad[int(np.sum(release_indices[:n])):int(np.sum(release_indices[:n]))+release_indices[n]] = grad_
            models[n] = model_

        ## dE/dc
        model = np.concatenate(models)
        grad[-numComponents:] = fmatrixT(model-dd, fs)
        return 2*grad

    c_0 = ics[:numComponents]/sum(ics[:numComponents])
    initial_conds_ = [0,]*numModels
    for n in range(numModels):
        p_ = exponentialModel(tranges[n],np.concatenate((c_0, timeconstants)) )
        convP = LinearOperator((len(p_), release_indices[n]), matvec=lambda x:pmatrix(x,p_,n), rmatvec = lambda x:pmatrixT(x,p_, n))
        opt_result = scipy.optimize.lsq_linear(convP, mu_vecs[n]*data[n])
        initial_conds_[n] = opt_result.x

    initial_cond = np.concatenate((np.concatenate(initial_conds_), c_0))


    bounds = ((0,None),)*(sum(release_indices) + numComponents)
    options = {'disp':verbose}
    if ftol is not None:
        options['ftol'] = ftol
    if gtol is not None:
        options['gtol'] = gtol
    opt_result = scipy.optimize.minimize(obj_fun, initial_cond, method=method, jac=jac, bounds=bounds, options=options)
    if verbose and not opt_result.success:
        print(opt_result.message)

    c_final = opt_result.x[-numComponents:]/np.sum(opt_result.x[-numComponents:])

    fs = [0,]*numModels
    ps = [0,]*numModels
    for n in range(numModels):
        f_ = np.zeros(len(tranges[n]))
        if n == 0:
            f_[:release_indices[n]] = opt_result.x[:release_indices[n]]*np.sum(opt_result.x[-numComponents:])
        else:
            f_[:release_indices[n]] = opt_result.x[int(np.sum(release_indices[:n])):int(np.sum(release_indices[:n]))+release_indices[n]]*np.sum(opt_result.x[-numComponents:])
        fs[n] = f_
        ps[n] = exponentialModel(tranges[n], np.concatenate((c_final, timeconstants)))

    return fs, ps, c_final, obj_fun(opt_result.x), jac(opt_result.x)

def blindDeconvN_Linear(tranges, data, release_indices, dt, ics, K = 100, thresh=1e-5, mu=1, scale_factor=1, dense=False, notebookMode = False):
    """Performs blind deconvolution of M eye position traces with a n-component multiexponential plant with unknown coefficients using alternating least squares,
    assuming applied force becomes zero at some point.

    Arguments:
        tranges, data:
            lists of M arrays containing time points and eye position, respectively
        release_indices:
            list containing index of time array after which applied force is zero, for each of the M inputs
        dt:
            time step size
        ics:
            array of n initial guesses for the plant coefficients, followed by corresponding known inverse plant time constants
        K:
            number of iterations of alternating least squares (1 iteration = one plant step and one force step)
        thresh:
            function terminates when change in const function is less than this value
        mu:
            scale of weighting value to compensate for M input traces having different lengths
        scale_factor:
            scaling applied to cost function
        dense:
            False to use sparse matrix methods
        notebookMode:
            True to use tqdm notebook widget

    Returns:
        list of M applied force traces,
        list of M plant time courses (length same as tranges[m]),
        coefficients of plant,
        force cost function and
        plant cost function at end of optimization, and
        gradients of both cost functions evaluated at end of optimization.
    """

    numComponents = len(ics)//2
    numModels = len(tranges)
    timeconstants = ics[-numComponents:]

    mu_vecs = [0,]*numModels
    dd = np.array([])
    for n in range(numModels):
        mu_vec = np.ones(len(tranges[n]))*np.sqrt(scale_factor/(len(tranges[n])-release_indices[n]))
        mu_vec[:release_indices[n]] = np.sqrt(mu*scale_factor/(release_indices[n]))
        mu_vecs[n] = mu_vec
        dd = np.concatenate((dd, mu_vec*data[n]))

    def pmatrix(x, p, num):
        mu_vec = mu_vecs[num]
        conv_p = dt*np.convolve(p, x, mode='full')
        return mu_vec*conv_p[:len(p)]
    def pmatrixT(x,p, num):
        mu_vec = mu_vecs[num]
        release_index = release_indices[num]
        corr_p = dt*np.correlate(mu_vec*x, p, mode='full')
        return corr_p[-len(x):-len(x)+release_index]

    def fmatrix(x, f):
        returnvec = np.array([])
        for n in range(numModels):
            e_ = exponentialModel(tranges[n], np.concatenate((x, timeconstants)))
            conv_f_ = dt*np.convolve(f[n], e_, mode='full')
            returnvec = np.concatenate((returnvec, mu_vecs[n]*conv_f_[:len(f[n])]))
        return returnvec
    def fmatrixT(x,f):
        outvec = np.zeros(numComponents)
        mu_vec = np.concatenate(mu_vecs)
        xx = mu_vec*x
        for i in range(numComponents):
            tempvec = np.array([])
            for n in range(numModels):
                tempvec_ = dt*np.convolve(f[n],np.exp(-tranges[n]*timeconstants[i]), mode='full')
                tempvec = np.concatenate((tempvec, tempvec_[:len(f[n])]))
            outvec[i] = np.dot(tempvec, xx)
        return outvec

    def jac(x, fs):
        grad = np.zeros(sum(release_indices) + numComponents)
        ps = [0,]*numModels
        models = [0,]*numModels
        for n in range(numModels):
            p_ = exponentialModel(tranges[n], np.concatenate((x[-numComponents:], timeconstants)))

            model_ = pmatrix(f_, p_, n)
            grad_ = pmatrixT(model_ - mu_vecs[n]*data[n], p_, n)
            if n == 0:
                grad[:release_indices[0]] = grad_
            else:
                grad[int(np.sum(release_indices[:n])):int(np.sum(release_indices[:n]))+release_indices[n]] = grad_
            models[n] = model_

        ## dE/dc
        model = np.concatenate(models)
        grad[-numComponents:] = fmatrixT(model-dd, fs)
        return 2*grad

    ## Find applied force
    def find_f(p,d,num):
        # Construct F^(k) matrix
        if dense:
            convP = dt*np.dot(np.diag(mu_vec),convMat(p, release_indices[num]))
            optresult = scipy.optimize.lsq_linear(convP,mu_vecs[num]*d, bounds=(0, np.inf), method='bvls')
        else:
            convP = LinearOperator((len(p), release_indices[num]), matvec=lambda x:pmatrix(x,p,num), rmatvec = lambda x:pmatrixT(x,p, num))
            optresult = scipy.optimize.lsq_linear(convP, mu_vecs[num]*d, bounds=(0, np.inf))

        return_f = np.zeros(len(p))
        return_f[:release_indices[num]] = optresult.x
        return optresult.success, return_f, optresult.cost*2

    ## Find filter estimate f
    def find_p(fs):
        if dense:
            As = [0,]*numModels
            for n in range(numModels):
                expmatrix_ = np.zeros((len(tranges[n]), numComponents))
                for i in range(numComponents):
                    expmatrix_[:,i] = np.exp(-tranges[n]*timeconstants[i])
                convF_ = dt*np.dot(np.diag(mu_vecs[n]), convMat(fs[n], len(fs[n])))
                As[n] = np.dot(convF_, expmatrix_)
            A = np.vstack(As)
            # p_coeffs, err = scipy.optimize.nnls(A, dd)
            optresult = scipy.optimize.lsq_linear(A, dd, bounds=(0, np.inf), method='bvls')
            succ = True
        else:
            len_f = sum([len(f) for f in fs])
            A = LinearOperator((len_f, numComponents), matvec=lambda x:fmatrix(x,fs), rmatvec = lambda x:fmatrixT(x,fs))
            optresult = scipy.optimize.lsq_linear(A, dd, bounds=(0, np.inf))
            if not optresult.success:
                print(optresult.message)
            p_coeffs = optresult.x
            err = optresult.cost*2
            succ = optresult.success
        ps = [0,]*numModels
        for n in range(numModels):
            ps[n] = exponentialModel(tranges[n], np.concatenate((p_coeffs, timeconstants)))

        return succ, ps, p_coeffs, err

    c_0 = ics[:numComponents]/sum(ics[:numComponents])

    ps = [0,]*numModels
    for n in range(numModels):
        ps[n] = exponentialModel(tranges[n], ics)
    fs = [0,]*numModels

    loss_p = np.zeros(K)
    loss_f = np.zeros((numModels, K))

    if notebookMode:
        counter = tqdm.notebook.trange
    else:
        counter = tqdm.trange

    for i in counter(K):
        for n in range(numModels):

            succ, f_, loss_f[n,i] = find_f(ps[n], data[n], n)
            if succ:
                fs[n] = f_
            else:
                i -= 1
                break

        succ, ps_, c_0_, loss_p[i] = find_p(fs)
        if succ:
            ps = ps_
            c_0 = c_0_
        else:
            i -= 1
            break
        for n in range(numModels):
            ps[n] /= np.sum(c_0)
            fs[n] *= np.sum(c_0)
        c_0 /= np.sum(c_0)
        if(i > 1 and 0 <= loss_p[i-1]-loss_p[i] < thresh):
            break
    current_grad = jac(c_0, fs)
    return fs, ps, c_0, loss_f[:,:i+1], loss_p[:i+1], current_grad
