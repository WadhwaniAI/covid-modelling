import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import copy
from main.seir.optimiser import Optimiser


def plot_sensitivity(predictions_dict, which_fit='m1', var_name=None, param_range=np.linspace(1, 100, 201),
                     comp_name='recovered', aux_comp='total_infected'):
    if var_name == None:
        var_name = 'T_recov_{}'.format(comp_name)
    variable_param_ranges = {
        var_name: param_range
    }

    default_params = copy.copy(predictions_dict[which_fit]['best_params'])
    df_train = copy.copy(predictions_dict[which_fit]['df_train'])
    df_val = copy.copy(predictions_dict[which_fit]['df_val'])
    run_params = copy.copy(predictions_dict[which_fit]['run_params'])

    train_period = run_params['train_period']
    val_period = run_params['val_period']
    model = run_params['model_class']
    loss_indices = [-train_period, None]

    if aux_comp == None:
        which_compartments = [comp_name]
    else:
        which_compartments = [comp_name, aux_comp]

    try:
        for key in variable_param_ranges.keys():
            del default_params[key]
    except Exception as err:
        print('')
    optimiser = Optimiser()
    extra_params = optimiser.init_default_params(df_train, N=1e7, initialisation='intermediate', 
                                                 train_period=train_period)
    default_params = {**default_params, **extra_params}
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days
    loss_array, params_dict = optimiser.gridsearch(df_train, default_params, variable_param_ranges, model=model, 
                                                   method='mape', loss_indices=loss_indices, 
                                                   which_compartments=which_compartments,
                                                   total_days=total_days, debug=False)

    return params_dict, loss_array

def sensitivity_all_comps(predictions_dict, which_fit='m1'):
    var_tuples = [
        ('lockdown_R0', np.linspace(1, 1.5, 101), 'total_infected', None),
        ('I_hosp_ratio', np.linspace(0, 1, 201), 'total_infected', None),
        ('E_hosp_ratio', np.linspace(0, 2, 201), 'total_infected', None),
        ('P_fatal', np.linspace(0, 1, 201), 'deceased', 'total_infected'),
        ('T_recov_severe', np.linspace(1, 100, 101), 'recovered', 'total_infected'),
        ('T_recov_fatal', np.linspace(1, 100, 101), 'deceased', 'total_infected')
    ]

    best_params = copy.copy(predictions_dict[which_fit]['best_params'])

    fig, axs = plt.subplots(figsize=(18, 12), nrows=3, ncols=2)
    fig.suptitle('Sensitivity of variables with respect to corresponding compartments')
    fig.tight_layout(pad=5.0)
    for i, ax in enumerate(axs.flat):
        var_name, param_range, comp_name, aux_comp = var_tuples[i]
        params_dict, loss_array = plot_sensitivity(predictions_dict, which_fit, var_name=var_name,
                                                   param_range=param_range, comp_name=comp_name, aux_comp=aux_comp)
        ax.plot([x[list(x.keys())[0]] for x in params_dict], loss_array)
        ax.set_ylabel('Loss Value (MAPE)')
        ax.set_xlabel('{}'.format(var_name))
        ax.set_title('MAPE for {} vs {} '.format(comp_name, var_name))
        ax.grid()
        best_params[var_name] = params_dict[np.argmin(loss_array)][var_name]

    optimiser = predictions_dict[which_fit]['optimiser']
    df_prediction = optimiser.solve(best_params, predictions_dict[which_fit]['default_params'], 
                                    predictions_dict[which_fit]['df_train'], 
                                    end_date=predictions_dict[which_fit]['df_val'].iloc[-1, :]['date'], 
                                    model=predictions_dict[which_fit]['run_params']['model_class'])
    return fig, best_params, df_prediction
