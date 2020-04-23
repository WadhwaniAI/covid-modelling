from curvefit.core.utils import data_translator
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter

def plot_draws_deriv(generator, prediction_times, draw_space, plot_obs, sharex=True, sharey=True, plot_uncertainty=True):
    fig, ax = plt.subplots(len(generator.groups), 1, figsize=(8, 4 * len(generator.groups)),
                               sharex=sharex, sharey=sharey)
    if len(generator.groups) == 1:
        ax = [ax]
    for i, group in enumerate(generator.groups):
        draws = generator.draws[group].copy()
        draws = data_translator(
            data=draws,
            input_space=generator.predict_space,
            output_space=draw_space
        )
        mean_fit = generator.mean_predictions[group].copy()
        mean_fit = data_translator(
            data=mean_fit,
            input_space=generator.predict_space,
            output_space=draw_space
        )
        mean = draws.mean(axis=0)
        ax[i].plot(prediction_times, mean, c='red', linestyle=':')
        ax[i].plot(prediction_times, mean_fit, c='black')

        if plot_uncertainty:
            lower = np.quantile(draws, axis=0, q=0.025)
            upper = np.quantile(draws, axis=0, q=0.975)
            ax[i].plot(prediction_times, lower, c='red', linestyle=':')
            ax[i].plot(prediction_times, upper, c='red', linestyle=':')

        if plot_obs is not None:
            df_data = generator.all_data.loc[generator.all_data[generator.col_group] == group].copy()
            ax[i].scatter(df_data[generator.col_t], df_data[plot_obs])

        ax[i].set_title(f"{group} predictions")

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def smooth(y, smoothing_window):
    box = np.ones(smoothing_window)/smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# def sma(y, smoothing_window):
#     return y.rolling(window=smoothing_window)

# def ema(y, smoothing_window):
#     return y.ewm(span=smoothing_window, adjust=False)

def setup_plt(ycol):
    register_matplotlib_converters()
    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d.%m"))
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel(ycol)