import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from utils.enums.columns import *

def plot_fit(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district,
                 which_compartments=['hospitalised', 'total_infected'], description='', savepath=None):
    """Helper function for creating plots for the training pipeline

    Arguments:
        df_prediction {pd.DataFrame} -- The prediction dataframe outputted by the model
        df_train {pd.DataFrame} -- The train dataset (with rolling average)
        df_val {pd.DataFrame} -- The val dataset (with rolling average)
        df_train_nora {pd.DataFrame} -- The train dataset (with no rolling average)
        df_val_nora {pd.DataFrame} -- The val dataset (with no rolling average)
        train_period {int} -- Length of train period
        state {str} -- Name of state
        district {str} -- Name of district

    Keyword Arguments:
        which_compartments {list} -- Which buckets to plot (default: {['hospitalised', 'total_infected']})
        description {str} -- Additional description for the plots (if any) (default: {''})

    Returns:
        ax -- Matplotlib ax object
    """
    # Create plots
    if isinstance(df_val, pd.DataFrame):
        df_true_plotting_rolling = pd.concat(
            [df_train, df_val], ignore_index=True)
        df_true_plotting = pd.concat(
            [df_train_nora, df_val_nora], ignore_index=True)
    else:
        df_true_plotting_rolling = df_train
        df_true_plotting = df_train_nora
    df_predicted_plotting = df_prediction.loc[df_prediction['date'].isin(
        df_true_plotting['date']), ['date']+which_compartments]

    fig, ax = plt.subplots(figsize=(12, 12))
    for compartment in compartments:
        if compartment.name in which_compartments:
            ax.plot(df_true_plotting[compartments[0].name], df_true_plotting[compartment.name],
                    '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
            ax.plot(df_true_plotting_rolling[compartments[0].name], df_true_plotting_rolling[compartment.name],
                    '-', color=compartment.color, label='{} (Obs RA)'.format(compartment.label))
            ax.plot(df_predicted_plotting[compartments[0].name], df_predicted_plotting[compartment.name],
                    '-.', color=compartment.color, label='{} (Predicted)'.format(compartment.label))
    

    ax.axvline(x=df_train.iloc[-train_period, :]['date'], ls='--', color='brown', label='Train starts')
    if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
        ax.axvline(x=df_val.iloc[0, ]['date'], ls='--', color='black', label='Val starts')
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.legend()
    plt.title('{} - ({} {})'.format(description, state, district))
    plt.grid()

    if savepath is not None:
        plt.savefig(savepath)
    return ax
