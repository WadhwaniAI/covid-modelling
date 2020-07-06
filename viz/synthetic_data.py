import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from datetime import timedelta

from utils.enums.columns import *
from viz.fit import axis_formatter


def plot_compartment_for_datasets(ax, df_true, df_prediction, df_train, district,
                                  s1_start, s2_start, s3_start, train_start, test_start,
                                  series_end, graph_start, graph_end,
                                  title=None, which_compartments=Columns.which_compartments()):
    """Plots ground truth data, synthetic data and predictions for a compartment

    Args:
        ax (axes.Axes): axes object of plot
        df_true (pd.DataFrame): Ground truth data
        df_prediction (pd.DataFrame): Predictions
        df_train (pd.DataFrame): Training data (custom dataset)
        district (str): name of district
        s1_start (str): start date of series s1
        s2_start (str): start date of series s2
        s3_start (str): start date of series s3
        train_start (str): start date of train split
        test_start (str): start date of test split
        series_end (str): last date of series s3
        graph_start (str): start date of graph
        graph_end (str): end date of graph
        title (str, optional): title for plots
        which_compartments (list(enum), optional): list of compartments plotted

    Returns:
        axes.Axes: axes object of plot
    """

    for col in Columns.which_compartments():
        if col in which_compartments:
            ax.plot(df_true['date'], df_true[col.name],
                    '-o', color=col.color, label=f'{col.label} (Observed)')
            ax.plot(df_train['date'], df_train[col.name],
                    'x', color=col.color, label=f'{col.label} (Train data)')
            ax.plot(df_prediction["date"], df_prediction[col.name],
                    '-', color=col.color, label=f'{col.label} (Forecast)')

            s1_start = pd.to_datetime(s1_start)
            s2_start = pd.to_datetime(s2_start)
            s3_start = pd.to_datetime(s3_start)
            series_end = pd.to_datetime(series_end)
            train_start = pd.to_datetime(train_start)
            test_start = pd.to_datetime(test_start)
            graph_start = pd.to_datetime(graph_start)
            graph_end = pd.to_datetime(graph_end)

            line_height = plt.ylim()[1]
            ax.plot([train_start, train_start], [0, line_height], '--', color='black', label='Train starts')
            ax.plot([test_start, test_start], [0, line_height], '--', color='black', label='Test starts')

            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            ax.axvspan(s1_start, s2_start, alpha=0.1, color='red', label='s1')
            ax.axvspan(s2_start, s3_start, alpha=0.1, color='yellow', label='s2')
            ax.axvspan(s3_start, series_end, alpha=0.1, color='green', label='s3')

            ax.legend(loc="upper left")
            ax.tick_params(labelrotation=45)
            ax.grid()

            ax.set_xlim(graph_start, graph_end)

            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('No of People', fontsize=10)

            ax.title.set_text('Forecast - {} - {}'.format(district, title))

    return ax


def plot_all_experiments(df_true, predictions_dict, district,
                         actual_start_date, allowance, s1, s2, s3, shift, train_period,
                         output_folder, titles=None):
    """Create plots for all buckets and all datasets

    Args:
        df_true (pd.DataFrame): ground truth data
        predictions_dict (dict): Dictionary of results dicts from SEIR c2 models
        district (str): name of district
        actual_start_date (datetime.datetime): date from which series s1 begins
        allowance (int): number of days of ground data before train split to prevent nan when rolling average is taken
        s1 (int): length of series s1
        s2 (int): length of series s2
        s3 (int): length of series s3
        shift (int): number of days of data removed at the beginning of the dataset
        train_period (int): train period
        output_folder (str): output folder path
        titles (list[str], optional): titles for plots
    """

    s1_start = actual_start_date.strftime("%m-%d-%Y")
    s2_start = (actual_start_date + timedelta(s1)).strftime("%m-%d-%Y")
    s3_start = (actual_start_date + timedelta(s1 + s2)).strftime("%m-%d-%Y")
    train_start = (actual_start_date + timedelta(s1 + s2 - train_period)).strftime("%m-%d-%Y")
    test_start = (actual_start_date + timedelta(s1 + s2)).strftime("%m-%d-%Y")
    series_end = (actual_start_date + timedelta(s1 + s2 + s3)).strftime("%m-%d-%Y")
    graph_start = (actual_start_date - timedelta(allowance + 1)).strftime("%m-%d-%Y")
    graph_end = (actual_start_date + timedelta(s1 + s2 + s3 + 1)).strftime("%m-%d-%Y")

    if titles is None:
        titles = ['Using ground truth data', 'Using data from IHME forecast', 'Using data from SEIR forecast']

    df_true = df_true.iloc[shift:, :].head(allowance + s1 + s2 + s3)

    which_compartments = Columns.which_compartments()

    for col in range(len(which_compartments)):
        fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(15, 15))
        for row in range(len(ax)):
            ax[row] = plot_compartment_for_datasets(ax[row], df_true, predictions_dict[row]['m1']['df_prediction'],
                                                    predictions_dict[row]['m1']['df_district'],
                                                    district, s1_start, s2_start, s3_start, train_start, test_start,
                                                    series_end, graph_start, graph_end, title=titles[row],
                                                    which_compartments=[which_compartments[col]])

            filename = output_folder + 'experiments_' + which_compartments[col].name
            fig.savefig(filename)
        plt.close()


def plot_fit_uncertainty(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, test_period,
                         state, district, draws=None, which_compartments=['hospitalised', 'total_infected'],
                         description='', savepath=None):
    # TODO: Use plot_fit from fit.py with modifications instead
    """Helper function for creating plots for the training pipeline

    Arguments:
        df_prediction {pd.DataFrame} -- The prediction dataframe outputted by the model
        df_train {pd.DataFrame} -- The train dataset (with rolling average)
        df_val {pd.DataFrame} -- The val dataset (with rolling average)
        df_train_nora {pd.DataFrame} -- The train dataset (with no rolling average)
        df_val_nora {pd.DataFrame} -- The val dataset (with no rolling average)
        train_period {int} -- Length of train period
        test_period {int} -- Length of test period
        state {str} -- Name of state
        district {str} -- Name of district

    Keyword Arguments:
        which_compartments {list} -- Which buckets to plot (default: {['hospitalised', 'total_infected']})
        description {str} -- Additional description for the plots (if any) (default: {''})
        savepath {str} -- Path to save plot

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
        df_true_plotting['date']), ['date'] + which_compartments]

    plot_ledger = {
        'base': True,
        'severity': True,
        'bed': True
    }
    for i, key in enumerate(plot_ledger.keys()):
        names = [x.name for x in compartments[key]]
        if np.sum(np.in1d(names, which_compartments)) == 0:
            plot_ledger[key] = False

    n_rows = np.sum(list(plot_ledger.values()))
    fig, axs = plt.subplots(nrows=n_rows, figsize=(12, 10 * n_rows))
    fig.suptitle('{} {}, {}'.format(description, district, state))
    i = 0
    for key in plot_ledger.keys():
        if not plot_ledger[key]:
            continue
        if n_rows > 1:
            ax = axs[i]
        else:
            ax = axs
        names = [x.name for x in compartments[key]]
        comp_subset = np.array(which_compartments)[np.in1d(which_compartments, names)]
        legend_elements = []
        for compartment in compartments[key]:
            if compartment.name in comp_subset:
                ax.plot(df_true_plotting[compartments['date'][0].name], df_true_plotting[compartment.name],
                        '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
                ax.plot(df_true_plotting_rolling[compartments['date'][0].name],
                        df_true_plotting_rolling[compartment.name],
                        '-', color=compartment.color, label='{} (Obs RA)'.format(compartment.label))
                ax.plot(df_predicted_plotting[compartments['date'][0].name], df_predicted_plotting[compartment.name],
                        '-.', color=compartment.color, label='{} (Predicted)'.format(compartment.label))
                if draws is not None:
                    ax.errorbar(df_prediction['date'][train_period:train_period+test_period],
                                df_prediction[compartment.name][train_period:train_period+test_period],
                                yerr=draws[compartment.name]['draws'][:, train_period:train_period+test_period],
                                lw=1.0, ls='-.', color='lightcoral', barsabove='False', label='draws')

                legend_elements.append(
                    Line2D([0], [0], color=compartment.color, label=compartment.label))

        ax.axvline(x=df_train.iloc[-train_period, :]['date'], ls=':', color='brown', label='Train starts')
        if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
            ax.axvline(x=df_val.iloc[0, ]['date'], ls=':', color='black', label='Val starts')

        axis_formatter(ax, legend_elements, custom_legend=False)
        i += 1

        ax.legend(loc="upper left")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.close()

    return fig


def plot_compartment_against_baseline(ax, df_true, df_prediction, df_prediction_baseline, df_train, district,
                                      s1_start, s2_start, s3_start, train_start, test_start, baseline_train_start,
                                      series_end, graph_start, graph_end,
                                      title=None, which_compartments=Columns.which_compartments()):
    """Plots ground truth data, synthetic data, predictions and baseline predictions for a compartment

    Args:
        ax (axes.Axes): axes object of plot
        df_true (pd.DataFrame): Ground truth data
        df_prediction (pd.DataFrame): Predictions
        df_prediction_baseline (pd.DataFrame): Baseline predictions
        df_train (pd.DataFrame): Training data (custom dataset)
        district (str): name of district
        s1_start (str): start date of series s1
        s2_start (str): start date of series s2
        s3_start (str): start date of series s3
        train_start (str): start date of train split
        test_start (str): start date of test split
        baseline_train_start (str): start date of baseline train split
        series_end (str): last date of series s3
        graph_start (str): start date of graph
        graph_end (str): end date of graph
        title (str, optional): title for plots
        which_compartments (list(enum), optional): list of compartments plotted

    Returns:
        axes.Axes: axes object of plot
    """

    for col in Columns.which_compartments():
        if col in which_compartments:
            ax.plot(df_true['date'], df_true[col.name],
                    '-o', color=col.color, label=f'{col.label} (Observed)')
            ax.plot(df_prediction_baseline['date'], df_prediction_baseline[col.name],
                    '-', color=col.color, label=f'{col.label} (Forecast)')
            ax.plot(df_prediction["date"], df_prediction[col.name],
                    '-.', color=col.color, label=f'{col.label} (Forecast with synthetic data)')
            ax.plot(df_train["date"], df_train[col.name],
                    'x', color=col.color, label=f'{col.label} (Train data)')

            s1_start = pd.to_datetime(s1_start)
            s2_start = pd.to_datetime(s2_start)
            s3_start = pd.to_datetime(s3_start)
            series_end = pd.to_datetime(series_end)
            train_start = pd.to_datetime(train_start)
            baseline_train_start = pd.to_datetime(baseline_train_start)
            test_start = pd.to_datetime(test_start)
            graph_start = pd.to_datetime(graph_start)
            graph_end = pd.to_datetime(graph_end)

            line_height = plt.ylim()[1]
            ax.plot([train_start, train_start], [0, line_height], '--', color='black', label='Train starts')
            ax.plot([test_start, test_start], [0, line_height], '--', color='black', label='Test starts')
            ax.plot([baseline_train_start, baseline_train_start], [0, line_height], '--', color='brown',
                    label='Baseline train starts')

            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            ax.axvspan(s1_start, s2_start, alpha=0.1, color='red', label='s1')
            ax.axvspan(s2_start, s3_start, alpha=0.1, color='yellow', label='s2')
            ax.axvspan(s3_start, series_end, alpha=0.1, color='green', label='s3')

            ax.legend(loc="upper left")
            ax.tick_params(labelrotation=45)
            ax.grid()

            ax.set_xlim(graph_start, graph_end)

            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('No of People', fontsize=10)

            ax.title.set_text('Forecast - {} - {}'.format(district, title))

    return ax


def plot_against_baseline(df_true, df_prediction, df_prediction_baseline, district,
                          actual_start_date, allowance, s1, s2, s3, shift, train_period, baseline_train_period,
                          output_folder, titles=None):
    """Create plots with baseline for all buckets and all synthetic datasets

    Args:
        df_true (pd.DataFrame): ground truth data
        df_prediction (dict): Dictionary of results dicts from SEIR c2 models
        df_prediction_baseline (dict): Results dict from SEIR baseline model
        district (str): name of district
        actual_start_date (datetime.datetime): date from which series s1 begins
        allowance (int): number of days of ground data before train split to prevent nan when rolling average is taken
        s1 (int): length of series s1
        s2 (int): length of series s2
        s3 (int): length of series s3
        shift (int): number of days of data removed at the beginning of the dataset
        train_period (int): train period
        baseline_train_period (int): baseline train period
        output_folder (str): output folder path
        titles (list[str], optional): titles for plots

    """

    s1_start = actual_start_date.strftime("%m-%d-%Y")
    s2_start = (actual_start_date + timedelta(s1)).strftime("%m-%d-%Y")
    s3_start = (actual_start_date + timedelta(s1 + s2)).strftime("%m-%d-%Y")
    train_start = (actual_start_date + timedelta(s1 + s2 - train_period)).strftime("%m-%d-%Y")
    baseline_train_start = (actual_start_date + timedelta(s1 - baseline_train_period)).strftime("%m-%d-%Y")
    test_start = (actual_start_date + timedelta(s1 + s2)).strftime("%m-%d-%Y")
    series_end = (actual_start_date + timedelta(s1 + s2 + s3)).strftime("%m-%d-%Y")
    graph_start = (actual_start_date - timedelta(allowance + 1)).strftime("%m-%d-%Y")
    graph_end = (actual_start_date + timedelta(s1 + s2 + s3 + 1)).strftime("%m-%d-%Y")

    if titles is None:
        titles = ['Using data from IHME forecast', 'Using data from SEIR forecast']

    df_true = df_true.iloc[shift:, :].head(allowance + s1 + s2 + s3)

    which_compartments = Columns.which_compartments()

    for col in range(len(which_compartments)):
        fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 15))
        for row in range(len(ax)):
            ax[row] = plot_compartment_against_baseline(ax[row], df_true, df_prediction[row+1]['m1']['df_prediction'],
                                                        df_prediction_baseline['m1']['df_prediction'],
                                                        df_prediction[row + 1]['m1']['df_district'],
                                                        district, s1_start, s2_start, s3_start, train_start, test_start,
                                                        baseline_train_start, series_end, graph_start, graph_end,
                                                        title=titles[row],
                                                        which_compartments=[which_compartments[col]])

            filename = output_folder + 'baseline_' + which_compartments[col].name
            fig.savefig(filename)
        plt.close()
