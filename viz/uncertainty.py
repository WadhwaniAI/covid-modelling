import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import datetime
import copy

from main.seir.forecast import get_forecast, order_trials, top_k_trials, forecast_k
from utils.enums import Columns, SEIRParams

def plot_ptiles(predictions_dict, ptile_pred_dict, train_fit='m2', vline=None,
        which_compartments=[Columns.active]):
    
    predictions = list(ptile_pred_dict.values())
    df_master = predictions[0]
    for df in predictions[1:]:
        df_master = pd.concat([df_master, df], ignore_index=True)
    df_true = predictions_dict[train_fit]['df_district']
    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name], df_true[compartment.name],
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        for i, (ptile, df_prediction) in enumerate(ptile_pred_dict.items()):
            sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                        ls='-', label=f'{compartment.label} Percentile :{ptile}')
            texts.append(plt.text(
                x=df_prediction[Columns.date.name].iloc[-1], 
                y=df_prediction[compartment.name].iloc[-1], s=ptile))
                
        if vline:
            plt.axvline(datetime.datetime.strptime(vline, '%Y-%m-%d'))
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.ylabel('No of People', fontsize=16)
        plt.yscale('log')
        plt.xlabel('Time', fontsize=16)
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.legend()
        plt.title('Percentile Forecasts - ({} {})'.format(predictions_dict['state'], predictions_dict['dist']), fontsize=16)
        plt.grid()
        plots[compartment] = ax
    return plots

