import numpy as np

import sys
sys.path.append('../../tools/')
import fitting_functions

import scipy.optimize
import tqdm
import scipy.io as sio

def importLongFixation(filename):
    data_file = sio.loadmat(filename)
    trange = data_file['trange'][0]
    fixation = data_file['fixation'][0]
    return trange, fixation

 __name__ == "__main__":
    num_ics = 100
    if len(sys.argv) == 2:
        num_ics = int(sys.argv[1])
    traces = ['090711e_0006', '090811c_0002', '090811d_0002', '090811d_0004','091111a_0001',
          '091111a_0003','091111c_0003','091211a_0002','091211a_0005']
    for fish_num in tqdm.trange(len(traces), desc='Trace no.:'):
        trange, fixation = importLongFixation('../../data/long-fixations/fixed/'+traces[fish_num]+'_long.mat')
        # Fit fixations that are length of average active fixation
        fits, lls, sses = fitting_functions.fitNEyePositions(trange[:300]-trange[0], np.reshape(fixation[:300], (1, 300)), max_num_components=6, num_ics=num_ics, inverse_tau_max = 1/(3*72*2e-4))
        sio.savemat('results/'+traces[fish_num]+'_long.mat', {'fits': fits, 'sses':sses, 'lls':lls}, appendmat=False)
