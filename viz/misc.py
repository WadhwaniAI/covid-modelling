import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime
import copy

from utils.generic.enums.columns import *
from viz.utils import axis_formatter


def compare_forecast(predictions_dict, df_true, which_forecast=80, truncate_pretrain_data=False, 
                     separate_compartments=False):
    df_prediction = copy.copy(predictions_dict['m2']['forecasts'][which_forecast])
    df_train = copy.copy(predictions_dict['m2']['df_train'])
    train_period = predictions_dict['m2']['run_params']['train_period']
    if truncate_pretrain_data:
        df_prediction = df_prediction.loc[(df_prediction['date'] > df_train.iloc[-train_period, :]['date']) &
                                          (df_prediction['date'] <= df_true.iloc[-1, :]['date'])]
        df_true = df_true.loc[df_true['date'] > df_train.iloc[-train_period, :]['date']]
        df_prediction.reset_index(inplace=True, drop=True)
        df_true.reset_index(inplace=True, drop=True)
    
    if separate_compartments:
        fig, axs = plt.subplots(figsize=(18, 12), nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots(figsize=(12, 12))

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
    for i, ax in enumerate(iterable_axes):
        ax.axvline(x=df_train.iloc[-train_period, :]['date'], ls=':', color='brown', label='Train starts')
        ax.axvline(x=df_train.iloc[-1, :]['date'], ls=':', color='black', label='Last data point seen by model')
        axis_formatter(ax, None, custom_legend=False)
    
    fig.suptitle(f'Predictions of {which_forecast} vs Ground Truth (Unseen Data)')
    plt.tight_layout()
