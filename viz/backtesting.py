import matplotlib as mpl
import pandas as pd
import copy
from datetime import timedelta

from viz.utils import setup_plt
from viz.utils import axis_formatter
from data.processing.processing import get_data
from utils.generic.enums.columns import *


def plot_backtest_seir(gt_data_source='athena', preds_source='filename', fname_format='old_output', filename=None, 
                       predictions_dict=None, which_forecast=80, truncate_plotting_range=False,
                       separate_compartments=False, dataloading_params={'state': 'Maharashtra', 'district': 'Mumbai'}):
    """Function of backtesting plotting

    Args:
        gt_data_source (str, optional): Ground Truth data source. Defaults to 'athena'.
        preds_source (str, optional): Source of predictions ('filename'/'pickle'). Defaults to 'filename'.
        fname_format (str, optional): Format of predictions fname ('old_output'/'new_deciles'). 
        Required if preds_source == 'filename'. Defaults to 'old_output'.
        filename (str, optional): path to predictions filename. 
        Required if preds_source == 'filename'. Defaults to None.
        predictions_dict (dict, optional): Predictions dict loaded from pickle file. 
        Required if preds_source == 'pickle'. Defaults to None.
        which_forecast (int, optional): Which forecast (50, 80, best, etc). Defaults to 80.
        truncate_plotting_range (bool, optional): If true, only plots those days 
        for which forecasting was done + training days. Defaults to False.
        separate_compartments (bool, optional): If true, creates subplot for each compartment. Defaults to False.
        dataloading_params (dict, optional): Dict corresponding to location where forecasting was done. 
        Defaults to {'state': 'Maharashtra', 'district': 'Mumbai'}.

    Raises:
        ValueError: Please give legal fname_format : old_output or new_deciles
        ValueError: Please give a predictions_dict input, current input is None
        ValueError: Please give legal preds_source : either filename or pickle

    Returns:
        mpl.Figure, pd.DataFrame, pd.DataFrame : matplotlib figure, gt df, prediction df (both truncated)
    """
    
    # Getting gt data
    df_true = get_data(gt_data_source, dataloading_params)

    # Setting train_period to None
    train_period = None
    if preds_source == 'filename':
        df = pd.read_csv(filename)
        if fname_format == 'old_output':
            cols_to_delete = [x for x in df.columns if ('max' in x) or ('min' in x)]
            df = df.drop(cols_to_delete, axis=1)
            df = df.drop(['current_active', 'current_icu', 'current_ventilator',
                          'icu_mean', 'hospitalized_mean'], axis=1)
            df.rename({'current_hospitalized': 'current_active'}, axis=1, inplace=True)
            df.columns = [x if 'current' not in x else x.replace(
                'current', 'true') for x in df.columns]
            df.columns = [x if 'mean' not in x else 'pred_' +
                          x.replace('_mean', '') for x in df.columns]


            df_prediction = df[df['which_forecast'] == str(float(which_forecast))]
            predicted_cols = [x for x in df_prediction.columns if 'pred_' in x]
            true_cols = [x for x in df_prediction.columns if 'true_' in x]
            df_prediction = df_prediction.dropna(subset=predicted_cols, how='any', axis=0)
            df_prediction = df_prediction.drop(true_cols, axis=1)
            df_prediction = df_prediction.rename({'predictionDate': 'date'}, axis=1)
            df_prediction['date'] = pd.to_datetime(df_prediction['date'], format='%Y-%m-%d')
            df_prediction.columns = [x.replace('pred_', '') for x in df_prediction.columns]


        elif fname_format == 'new_deciles':
            multi_index = list(zip(df.loc[[0, 2], :].to_numpy().tolist()[0], 
                                   df.loc[[0, 2], :].to_numpy().tolist()[1]))
            multi_index[0] = ('date', 'date')
            for i, (percentile, column) in enumerate(multi_index):
                if column == 'total cases':
                    multi_index[i] = (percentile, 'total')
                if percentile != 'date':
                    multi_index[i] = (float(percentile), multi_index[i][1])
            df.columns = pd.MultiIndex.from_tuples(multi_index)
            df.drop([0, 1, 2], axis=0, inplace=True)
            df.reset_index(inplace=True, drop=True)
            df.loc[:, ('date', 'date')] = pd.to_datetime(df['date']['date'], 
                                                         format='%d/%m/%y')

            df_prediction = df[['date', which_forecast]]
            df_prediction.columns = [x[1] for x in df_prediction.columns]
            numeric_cols = ['total', 'active', 'recovered', 'deceased']
            df_prediction.loc[:, numeric_cols] = df_prediction.loc[:, numeric_cols].apply(pd.to_numeric)
        else:
            raise ValueError('Please give legal fname_format : old_output or new_deciles')
    elif preds_source == 'pickle':
        if predictions_dict is None:
            raise ValueError('Please give a predictions_dict input, current input is None')

        df_prediction = copy.copy(
            predictions_dict['forecasts'][which_forecast])
        df_train = copy.copy(predictions_dict['df_train'])
        train_period = predictions_dict['run_params']['split']['train_period']
    else:
        raise ValueError('Please give legal preds_source : either filename or pickle')

    if truncate_plotting_range:
        df_prediction = df_prediction.loc[(
            df_prediction['date'] <= df_true.iloc[-1, :]['date'])]
        df_true = df_true.loc[df_true['date']
                              >= (df_prediction.iloc[0, :]['date'] - timedelta(days=1))]
        df_true = df_true.loc[df_true['date']
                              <= df_prediction.iloc[-1, :]['date']]
        df_prediction.reset_index(inplace=True, drop=True)
        df_true.reset_index(inplace=True, drop=True)


    if separate_compartments:
        fig, axs = plt.subplots(figsize=(21, 12), nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots(figsize=(14, 12))

    for i, compartment in enumerate(compartments['base']):
        if separate_compartments:
            ax = axs.flat[i]
        ax.plot(df_true[compartments['date'][0].name].to_numpy(),
                df_true[compartment.name].to_numpy(),
                '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
        ax.plot(df_prediction[compartments['date'][0].name].to_numpy(),
                df_prediction[compartment.name].to_numpy(),
                '-.', color=compartment.color, label='{} (Predicted)'.format(compartment.label))

    if separate_compartments:
        iterable_axes = axs.flat
    else:
        iterable_axes = [ax]
    if train_period is None:
        train_period = 21
    for i, ax in enumerate(iterable_axes):
        ax.axvline(x=df_true.iloc[0, :]['date'],
                   ls=':', color='brown', label='Train starts')
        ax.axvline(x=df_true.iloc[train_period-1, :]['date'], ls=':',
                   color='black', label='Last data point seen by model')
        axis_formatter(ax, None, custom_legend=False)

    fig.suptitle(
        f'Predictions of {which_forecast} vs Ground Truth (Unseen Data)')
    fig.subplots_adjust(top=0.96)
    return fig, df_true.iloc[train_period:, :], df_prediction.iloc[train_period-1:, :]

def plot_backtest(results, data, dist, which_compartments=Columns.which_compartments(), 
                  scoring='mape', dtp=None, axis_name='No. People', savepath=None):
    title = f'{dist}' +  ' backtesting'
    # plot predictions against actual
    setup_plt(axis_name)
    plt.yscale("linear")
    plt.title(title)
    def div(series):
        if scoring=='mape':
            return series/100
        return series
    
    fig, ax = plt.subplots(figsize=(12, 12))
    # plot predictions
    cmap = mpl.cm.get_cmap('winter')
    for col in which_compartments:
        for i, run_day in enumerate(results.keys()):
            run_dict = results[run_day]
            preds = run_dict['df_prediction'].set_index('date')
            val_dates = run_dict['df_val']['date'] if run_dict['df_val'] is not None and len(run_dict['df_val']) > 0 else None
            errkey = 'val' if val_dates is not None else 'train'
            fit_dates = [n for n in run_dict['df_train']['date'] if n in preds.index]
            # fit_dates = run_dict['df_train']['date']
            
            color = cmap(i/len(results.keys()))
            ax.plot(fit_dates, preds.loc[fit_dates, col.name], ls='solid', c=color)
            ax.errorbar(fit_dates, preds.loc[fit_dates, col.name],
                yerr=preds.loc[fit_dates, col.name]*(div(run_dict['df_loss'].loc[col.name, errkey])), lw=0.5,
                color='lightcoral', barsabove='False', label=scoring)
            if val_dates is not None:
                ax.plot(val_dates, preds.loc[val_dates, col.name], ls='dashed', c=color,
                    label=f'run day: {run_day}')
                ax.errorbar(val_dates, preds.loc[val_dates, col.name],
                    yerr=preds.loc[val_dates, col.name]*(div(run_dict['df_loss'].loc[col.name, errkey])), lw=0.5,
                    color='lightcoral', barsabove='False', label=scoring)
                
            
        # plot data we fit on
        ax.scatter(data['date'].values, data[col.name].values, c='crimson', marker='+', label='data')
        plt.text(x=data['date'].iloc[-1], y=data[col.name].iloc[-1], s=col.name)
        
    # plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
        plt.clf()
    return

def plot_backtest_errors(results, data, file_prefix, which_compartments=Columns.which_compartments(), 
                         scoring='mape', savepath=None):
    start = data['date'].min()
    
    title = f'{file_prefix}' +  ' backtesting errors'
    errkey = 'val'

    setup_plt(scoring)
    plt.yscale("linear")
    plt.title(title)

    # plot error
    for col in which_compartments:
        ycol = col.name
        dates = [start + timedelta(days=run_day) for run_day in results.keys()]
        errs = [results[run_day]['df_loss'].loc[ycol, errkey] for run_day in results.keys()]
        plt.plot(dates, errs, ls='-', c='crimson',
            label=scoring)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
        plt.clf()
    return
