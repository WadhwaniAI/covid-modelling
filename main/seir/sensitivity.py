import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import copy

def gridsearch_single_param(predictions_dict, config, which_fit='m1', var_name=None,
                            param_range=np.linspace(1, 100, 201), df_train=None, train_period=None, 
                            comp_name='recovered', aux_comp='total', debug=False):
    if var_name == None:
        var_name = 'T_recov_{}'.format(comp_name)
    variable_param_ranges = {
        var_name: param_range
    }

    if not isinstance(df_train, pd.DataFrame):
        df_train = copy.copy(predictions_dict[which_fit]['df_train'])

    default_params = copy.copy(predictions_dict[which_fit]['best_params'])
    run_params = copy.copy(predictions_dict[which_fit]['run_params'])

    model = run_params['model_class']
    if train_period == None:
        train_period = run_params['split']['train_period']
    loss_indices = [-train_period, None]

    if aux_comp == 'all':
        loss_compartments = ['recovered', 'deceased', 'active', 'total']
    elif aux_comp == None:
        loss_compartments = [comp_name]
    else:
        loss_compartments = [comp_name, aux_comp]
    try:
        for key in variable_param_ranges.keys():
            del default_params[key]
    except Exception as err:
        print('')
    optimiser = predictions_dict[which_fit]['optimiser']
    extra_params = optimiser.init_default_params(df_train, config['fitting']['default_params'], 
                                                 train_period=train_period)
    default_params = {**default_params, **extra_params}
    loss_array, params_dict = optimiser.gridsearch(df_train, default_params, variable_param_ranges, model=model, 
                                                   loss_method='mape', loss_indices=loss_indices,
                                                   loss_compartments=loss_compartments, debug=debug)

    return params_dict, loss_array

def calculate_sensitivity_and_plot(predictions_dict, config, which_fit='m1'):
    best_params = copy.copy(predictions_dict[which_fit]['best_params'])

    config_sensitivity = copy.deepcopy(config['sensitivity'])
    for key, value in config_sensitivity.items():
        config_sensitivity[key][0] = np.linspace(value[0][0][0], value[0][0][1], value[0][1])

    var_tuples = [[key]+value for key, value in config_sensitivity.items()]

    nrows = int(round(len(var_tuples)/2+0.01))
    fig, axs = plt.subplots(figsize=(18, 4*nrows), nrows=nrows, ncols=2)
    fig.suptitle('Sensitivity of variables with respect to corresponding compartments')
    fig.tight_layout(pad=5.0)
    for i, ax in enumerate(axs.flat):
        var_name, param_range, comp_name, aux_comp = var_tuples[i]
        params_dict, loss_array = gridsearch_single_param(predictions_dict, config, which_fit, var_name=var_name,
                                                          param_range=param_range, comp_name=comp_name, 
                                                          aux_comp=aux_comp,debug=True)
        ax.plot([x[list(x.keys())[0]] for x in params_dict], loss_array)
        ax.set_ylabel('Loss Value (MAPE)')
        ax.set_xlabel('{}'.format(var_name))
        ax.set_title('MAPE for {} vs {} '.format(comp_name, var_name))
        ax.grid()
        best_params[var_name] = params_dict[np.argmin(loss_array)][var_name]

    optimiser = predictions_dict[which_fit]['optimiser']
    df_prediction = optimiser.solve({**best_params, **predictions_dict[which_fit]['default_params']}, 
                                    end_date=predictions_dict[which_fit]['df_district'].iloc[-1, :]['date'],
                                    model=predictions_dict[which_fit]['run_params']['model_class'])
    return fig, best_params, df_prediction


def plot_single_comp(params_dict, loss_array, comp_name='total'):
    fig, ax = plt.subplots(figsize=(12, 12))
    var_name = list(params_dict[0].keys())[0]
    ax.plot([x[list(x.keys())[0]] for x in params_dict], loss_array)
    ax.set_ylabel('Loss Value (MAPE)')
    ax.set_xlabel('{}'.format(var_name))
    ax.set_title('MAPE for {} vs {} '.format(comp_name, var_name))
    ax.grid()
    return fig
