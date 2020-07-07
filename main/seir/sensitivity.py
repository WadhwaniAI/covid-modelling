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
    best_params = params_dict[np.argmin(loss_array)]
    if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_val.iloc[-1, :]['date'], 
                                        model=model)
    else:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_train.iloc[-1, :]['date'],
                                        model=model)

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.plot([x[list(x.keys())[0]] for x in params_dict], loss_array)
    plt.ylabel('Loss Value (MAPE)')
    plt.xlabel('{}'.format(var_name))
    plt.title('MAPE for {} compartment vs {} '.format(comp_name, var_name))
    plt.grid()
    return fig, zip(params_dict, loss_array)
