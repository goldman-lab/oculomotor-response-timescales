import numpy as np

import sys
sys.path.append('../../../tools/')
import fitting_functions

import os

import scipy.io as sio
import scipy.optimize
import tqdm

def deconvolveEyePos(trange, eye_pos, plant, ind=-1):
    plant_model = fitting_functions.exponentialModel(trange, plant)
    dt = trange[2]-trange[1]

    def convMat(signal, m):
        mat = np.zeros((len(signal), m))
        mat[:,0] = signal
        for i in range(1, m):
            mat[i:,i] = signal[:len(signal)-i]
        return mat

    if ind == -1:
        ind = len(trange)
    mat = dt*convMat(plant_model, ind)
    optresult = scipy.optimize.lsq_linear(mat, eye_pos, bounds=(0, np.inf))
    f_ = np.zeros(len(trange))
    f_[:len(optresult.x)] = optresult.x
    return f_

if __name__ == "__main__":
    file_names = [ ('090711e_0006',), ('090811c_0002',), ('090811d_0002','090811d_0004',),
         ('091111a_0001', '091111a_0003'), ('091111c_0003',), ('091211a_0002', '091211a_0005')]
    best_num_components = {'090711e':3, '090811c':3, '090811d':3, '091111a':4, '091111c':3, '091211a':3}
    T_start = 17 # ~130 ms

    if not os.path.isdir('deconv'):
        os.makedirs('deconv')
    if not os.path.isdir('deconv/distributed'):
        os.makedirs('deconv/distributed')
    if not os.path.isdir('deconv/fast'):
        os.makedirs('deconv/fast')
    for fish_num in tqdm.trange(len(file_names), desc='Fish'):
        fish_name = file_names[fish_num][0][:-5]
        n = best_num_components[fish_name]

        # Load plants
        plant_file = sio.loadmat('../plants/best/distributed/'+fish_name+'.mat')
        plant = plant_file['plant'][0]

        plant_file = sio.loadmat('../plants/best/fast/'+fish_name+'.mat')
        plant_fast = plant_file['plant'][0]

        for trace_num in range(len(file_names[fish_num])):
            saccade_data_file = sio.loadmat('fit/'+file_names[fish_num][trace_num]+'.mat')
            trange_sacc = saccade_data_file['trange'][0]
            eye_pos_sacc = saccade_data_file['model'][0]
            drive = deconvolveEyePos(trange_sacc, eye_pos_sacc, plant)
            drive_fast = deconvolveEyePos(trange_sacc, eye_pos_sacc, fast_plant)
            sio.savemat('deconv/distributed/'+file_names[fish_num][trace_num]+'.mat', {'drive': drive})
            sio.savemat('deconv/fast/'+file_names[fish_num][trace_num]+'.mat', {'drive': drive_fast})
