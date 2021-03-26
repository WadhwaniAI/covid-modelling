import copy
import pickle as pkl
import sys

sys.path.append('../../')

from main.seir.main import single_fitting_cycle
from utils.generic.config import read_config

config_filenames = ['exp_simulate_3.yaml']
train_periods = [5,11,17,23,28]
PD = {}

for i,config_filename in enumerate(config_filenames):
    config = read_config(config_filename)
    PD[i] = {}
    x = 1
    for j,train_period in enumerate(train_periods):
        
        config['fitting']['split']['train_period'] = train_period
        print(config_filename,'-RUN------------------------->',x," ...Period-",config['fitting']['split']['train_period'])
        x = x+1
        PD[i][j] = single_fitting_cycle(**copy.deepcopy(config['fitting']))                                          

with open('../../misc/predictions/PD_beta_TP.pickle', 'wb') as handle:
    pkl.dump(PD, handle)
