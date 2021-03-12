import pandas as pd
import numpy as np
from copy import copy

import numpy as np
import pandas as pd
from curvefit.core.utils import data_translator
from curvefit.pipelines.basic_model import BasicModel

import sys

sys.path.append('../../')

from models.model import Model


class IHME(Model):

    def __init__(self, model_parameters: dict):
        """
        This model uses the curvefit module published by IHME (Wadhwani AI fork) to
        find an optimal param set for the specified function based on
        the initialization values and bounds specified

        This curve is then used to generate predictions.

        model_parameters must contain:
            xcol: str where df[xcol] is of type int, number of days from the beginning
            ycol: str where df[ycol] is the y to fit to, could be # cases, log(mortality), etc
            func: function to fit to
            date: str - column with date
            groupcol: str - column with group (state, etc) if wanting to fit to multiple groups
            priors: {
                'fe_init': list - required
                'fe_bounds': list - optional
                # see curvefit module docs for more priors
            }
            pipeline_args: {
                args to be passed into pipeline.run()
                n_draws, cv_threshold, smoothed_radius, num_smooths, 
                exclude_groups, exclude_below, exp_smoothing, max_last
            }
            covs: list, covariate column names
            predict_space: function to predict in (optional, default to func)

        The link functions are fixed at
            alpha: exp
            beta: identity
            p: exp
        ALl var link functions are set to identity

        Args:
            model_parameters (dict): model parameters dict, as specified above
        """
        self.model_parameters = model_parameters
        self.xcol = model_parameters.get('xcol')
        self.ycol = model_parameters.get('ycol')
        self.func = model_parameters.get('func')
        self.date = model_parameters.get('date')
        self.groupcol = model_parameters.get('groupcol')
        self.priors = model_parameters.get('priors')
        self.pipeline_args = model_parameters.get('pipeline_args')
        self.covs = model_parameters.get('covs')

        self.param_names = ['alpha', 'beta', 'p']
        self.predict_space = model_parameters.get('predict_space') if model_parameters.get(
            'predict_space') else self.func

        # link functions
        identity_fun = lambda x: x
        exp_fun = lambda x: np.exp(x)
        self.link_fun = [exp_fun, identity_fun, exp_fun]
        self.var_link_fun = [identity_fun, identity_fun, identity_fun]

        self.pipeline = None

    def predict(self, start_date, end_date, **kwargs):
        """[summary]

        Args:
            start_date (Timestamp): date to start predictions
            end_date (Timestamp): date to end predictions

        Returns:
            pd.DataFrame: predictions
        """
        data = self.pipeline.all_data.set_index('date')
        day0 = data.loc[data.index[0], 'day']
        start = int(day0) + (start_date - data.index[0]).days
        n_days = (end_date - start_date).days

        predictx = np.array([x for x in range(start, 1 + start + n_days)])
        self.run(predictx)
        return self.pipeline.predict(times=predictx, predict_space=self.predict_space, predict_group='all')

    def fit(self, data: pd.DataFrame):
        """
        Creates the underlying model with data as input

        Args:
            data (pd.DataFrame): dataframe to train model with, 
                containing columns specified in init function parameters
        """
        self.pipeline = BasicModel(
            all_data=data,  #: (pd.DataFrame) of *all* the data that will go into this modeling pipeline
            col_t=self.xcol,  #: (str) name of the column with time
            col_group=self.groupcol,  #: (str) name of the column with the group in it
            col_obs=self.ycol,  #: (str) the name of the column with observations for fitting the model
            col_obs_compare=self.ycol,
            # TODO: (str) the name of the column that will be used for predictive validity comparison
            all_cov_names=self.covs,
            # TODO: List[str] list of name(s) of covariate(s). Not the same as the covariate specifications
            fun=self.func,  #: (callable) the space to fit in, one of curvefit.functions
            predict_space=self.predict_space,
            # TODO confirm: (callable) the space to do predictive validity in, one of curvefit.functions
            obs_se_func=None,
            # TODO if we want to specify: (optional) function to get observation standard error from col_t
            fit_dict=self.priors,  #: keyword arguments to CurveModel.fit_params()
            basic_model_dict={  #: additional keyword arguments to the CurveModel class
                'col_obs_se': None,  # (str) of observation standard error
                'col_covs': [[cov] for cov in self.covs],
                # TODO: List[str] list of names of covariates to put on the parameters
                'param_names': self.param_names,  # (list{str}):
                'link_fun': self.link_fun,  # (list{function}):
                'var_link_fun': self.var_link_fun,  # (list{function}):
            },
        )

    def run(self, predictx, pipeline_args=None):
        """
        Function to actually fit the model, requires prediction dates

        Args:
            predictx (np.array): xcol values
            pipeline_args (dict, optional): additional args to override those specified in init.
                Defaults to None.

        Returns:
            BasicModel: instance of the curvefit model, fitted.
        """
        p_args = self.pipeline_args
        if pipeline_args is not None:
            p_args.update(pipeline_args)

        # pipeline
        self.pipeline.setup_pipeline()
        self.pipeline.run(n_draws=p_args['n_draws'], prediction_times=predictx,
                          cv_threshold=p_args['cv_threshold'], smoothed_radius=p_args['smoothed_radius'],
                          num_smooths=p_args['num_smooths'], exclude_groups=p_args['exclude_groups'],
                          exclude_below=p_args['exclude_below'], exp_smoothing=p_args['exp_smoothing'],
                          max_last=p_args['max_last']
                          )
        return self.pipeline

    def calc_draws(self):
        """
        Retrieves the draws for all groups

        Returns:
            [dict]: dict containing lower and upper (2.5% and 97.5%) percentiles
        """
        draws_dict = {}
        for group in self.pipeline.groups:
            draws = self.pipeline.draws[group].copy()
            draws = data_translator(
                data=draws,
                input_space=self.pipeline.predict_space,
                output_space=self.predict_space
            )
            mean_fit = self.pipeline.mean_predictions[group].copy()
            mean_fit = data_translator(
                data=mean_fit,
                input_space=self.pipeline.predict_space,
                output_space=self.predict_space
            )
            # mean = draws.mean(axis=0)

            lower = np.quantile(draws, axis=0, q=0.025)
            upper = np.quantile(draws, axis=0, q=0.975)
            draws_dict[group] = {
                'lower': lower,
                'upper': upper
            }
        return draws_dict

    def generate(self):
        """
        generates an untrained copy of the model

        Returns:
            self: copy of the untrained model 
        """
        out = IHME(self.model_parameters)
        out.xcol = self.xcol
        out.ycol = self.ycol
        out.func = self.func
        out.date = self.date
        out.groupcol = self.groupcol
        out.priors = copy(self.priors)
        out.pipeline_args = copy(self.pipeline_args)
        out.covs = copy(self.covs)

        out.param_names = copy(self.param_names)
        out.predict_space = self.predict_space

        out.link_fun = copy(self.link_fun)
        out.var_link_fun = copy(self.var_link_fun)

        out.pipeline = None
        return out
