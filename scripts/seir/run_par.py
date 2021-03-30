import copy

import sys
sys.path.append('../../')


from main.seir.main import single_fitting_cycle
from utils.generic.config import read_config

predictions_dict = {}
config_filename = 'exp_simulate_1.yaml'
print(config_filename)
config = read_config(config_filename)

predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))                                           

import pickle as pkl
with open('../../misc/predictions/exp_simulate_1_70.pickle', 'wb') as handle:
    pkl.dump(predictions_dict, handle)


