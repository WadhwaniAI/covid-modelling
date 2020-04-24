import matplotlib.pyplot as plt
from util import plot_draws_deriv, mape, setup_plt
from curvefit.core.functions import *
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_log_error

class Plotter():
    def __init__(self, pipeline, params, predictdate, predictx, file_prefix, output_folder, ycol, func):
        self.pipeline = pipeline
        self.output_folder = output_folder
        self.predictdate = predictdate
        self.predictx = predictx
        self.file_prefix = file_prefix
        self.ycol = ycol
        self.func = func
        self.derivs = {
            erf: derf,
            # gaussian_cdf: gaussian_pdf,
            log_erf: log_derf,
        }
        self.xcol = params.xcol
        self.groupcol = params.groupcol
        self.ycol = self.pipeline.col_obs
        self.data = self.pipeline.all_data
        self.date = params.date

    def plot_draws(self, dailycolname=None):
        # plot draws
        self.pipeline.plot_results(prediction_times=self.predictx)
        plt.savefig(f'{self.output_folder}/{self.file_prefix}_draws_{self.ycol}_{self.func.__name__}.png')
        plt.clf()

        if dailycolname:
            # plot draws - daily data/preds
            plot_draws_deriv(self.pipeline, self.predictx, self.derivs[self.func], dailycolname)
            plt.savefig(f'{self.output_folder}/{self.file_prefix}_draws_{self.ycol}_{self.derivs[self.func].__name__}.png')
            plt.clf()

    def predict(self, predict_space, multigroup):
        # pipeline.predict
        if multigroup:
            # for each groups if multiple groups, then aggregate
            # TODO: add an option for plotting per group; then we can just predict group='all'
                # and set a boolean check below to not plot group_predictions data when not wanted
            group_predictions = pd.DataFrame()
            for grp in self.data[self.groupcol].unique():
                grp_df = pd.DataFrame(columns=[self.xcol, self.date, self.groupcol, f'{self.ycol}_pred'])
                grp_df[f'{self.ycol}_pred'] = pd.Series(self.pipeline.predict(times=self.predictx, predict_space=predict_space, predict_group=grp))
                grp_df[self.groupcol] = grp
                grp_df[self.xcol] = self.predictx
                grp_df[self.date] = self.predictdate
                group_predictions = group_predictions.append(grp_df)
            predictions = group_predictions.groupby(self.xcol).sum()[f'{self.ycol}_pred']
        else:
            # otherwise just call predict once on all
            group_predictions = None
            predictions = self.pipeline.predict(times=self.predictx, predict_space=predict_space, predict_group='all')
        return group_predictions, predictions

    def calc_error(self, test, predictions, agg_data, daysback):
        # evaluate against test set - this is only done overall, not per group
        xtest, ytest = test[self.xcol], test[self.ycol]
        predtest = self.pipeline.predict(times=xtest, predict_space=self.func, predict_group='all')
        r2, msle = r2_score(ytest, predtest), None
        # this throws an error otherwise, hence the check
        if 'log' not in self.func.__name__ :
            msle = mean_squared_log_error(ytest, predtest)
        maperr_test = mape(ytest, predtest)
        print ('test set - mape: {} r2: {} msle: {}'.format(maperr_test, r2, msle))

        # evaluate overall - this is only done overall, not per group
        r2, msle = r2_score(agg_data[self.ycol], predictions[daysback:daysback+len(agg_data[self.ycol])]), None
        # this throws an error otherwise, hence the check
        if 'log' not in self.func.__name__ :
            msle = mean_squared_log_error(agg_data[self.ycol], predictions[daysback:daysback+len(agg_data[self.ycol])])
        maperr_overall = mape(agg_data[self.ycol], predictions[daysback:daysback+len(agg_data[self.ycol])])
        print ('overall - mape: {} r2: {} msle: {}'.format(maperr_overall, r2, msle))
        return maperr_test

    def plot_predictions(self, df, agg_data, agg_test, orig_ycol, test, daysback, smoothing_window=False, multigroup=False, dailycolname=None):
        group_predictions, predictions = self.predict(self.func, multigroup)
        maperr = self.calc_error(test, predictions, agg_data, daysback)
        title = f'{self.file_prefix} {self.ycol}' +  'fit to {}'
        # plot predictions against actual
        # set up the canvas
        setup_plt(self.ycol)
        plt.title(title.format(self.func.__name__))
        # actually plot the data
        if smoothing_window:
            # plot original data
            plt.plot(agg_data[self.date], agg_data[orig_ycol], 'k+', label='data (test)')
            plt.plot(agg_test[self.date], agg_test[orig_ycol], 'k+', label='data (test)')
        # plot data we fit on (smoothed if -s)
        plt.plot(agg_data[self.date], agg_data[self.ycol], 'r+', label='data')
        plt.plot(agg_test[self.date], agg_test[self.ycol], 'g+', label='data (test)')
        # plot predictions
        plt.plot(self.predictdate, predictions, 'r-', label='fit: {}: {}'.format(self.func.__name__, self.pipeline.mod.params))
        # plot error bars based on MAPE
        plt.errorbar(self.predictdate[df[self.date].nunique():], predictions[df[self.date].nunique():], yerr=predictions[df[self.date].nunique():]*maperr, color='black', barsabove='False')
        # plot each group's curve
        clrs = ['c', 'm', 'y', 'k']
        if len(self.data[self.groupcol].unique()) > 1:
            for i, grp in enumerate(self.data[self.groupcol].unique()):
                # plot each group's predictions
                plt.plot(self.predictdate, group_predictions[group_predictions[self.groupcol] == grp][f'{self.ycol}_pred'], f'{clrs[i]}-', label=grp)
                # plot each group's actual data
                plt.plot(self.data[self.data[self.groupcol] == grp][self.date], self.data[self.data[self.groupcol] == grp][self.ycol], f'{clrs[i]}+', label='data')
                plt.plot(test[test[self.groupcol] == grp][self.date], test[test[self.groupcol] == grp][self.ycol], f'{clrs[i]}+')
        
        plt.legend() 
        plt.savefig(f'{self.output_folder}/{self.file_prefix}_{self.ycol}_{self.func.__name__}.png')
        plt.clf()

        if dailycolname:
            # also predict daily numbers
            daily_group_predictions, daily_predictions = self.predict(self.derivs[self.func], multigroup)
            
            # calculate mape for error bars
            maperr = mape(agg_data[dailycolname], daily_predictions[daysback:daysback+len(agg_data[dailycolname])])
            print(f"Daily MAPE: {maperr}")
            
            # plot daily predictions against actual
            setup_plt(self.ycol)
            plt.title(title.format(self.derivs[self.func].__name__))
            # plot daily deaths
            plt.plot(agg_data[self.date], agg_data[dailycolname], 'b+', label='data')
            plt.plot(agg_test[self.date], agg_test[dailycolname], 'g+', label='data (test)')
            # against predicted daily deaths
            plt.plot(self.predictdate, daily_predictions, 'r-', label='fit: {}: {}'.format(self.derivs[self.func].__name__, self.pipeline.mod.params))
            # with error bars
            plt.errorbar(self.predictdate[df[self.date].nunique():], daily_predictions[df[self.date].nunique():], yerr=daily_predictions[df[self.date].nunique():]*maperr, color='black', barsabove='False')
            # including per group predictions if multiple
            # TODO: add per group observed data here
            if len(self.data[self.groupcol].unique()) > 1:
                for i, grp in enumerate(self.data[self.groupcol].unique()):
                    plt.plot(self.predictdate, daily_predictions[daily_predictions[self.groupcol] == grp][f'{self.ycol}_pred'], f'{clrs[i]}-', label=grp)
            
            plt.legend() 
            plt.savefig(f'{self.output_folder}/{self.file_prefix}_{self.ycol}_{self.derivs[self.func].__name__}.png')
            # plt.show() 
            plt.clf()