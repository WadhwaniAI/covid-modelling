# Model Reference Links

[IMHE CurveFit Documentation](https://ihmeuw-msca.github.io/CurveFit/)

[IMHE CurveFit Code Repo](https://github.com/ihmeuw-msca/CurveFit)

[IMHE paper](https://www.medrxiv.org/content/10.1101/2020.03.27.20043752v1.full.pdf)

[IMHE technical writeup](http://www.healthdata.org/sites/default/files/files/Projects/COVID/RA_COVID-forecasting-USA-EEA_042120.pdf)

[IMHE dashboard - US/Europe](https://covid19.healthdata.org/united-states-of-america)

# Future explorations
- BasicModel versus the other defined (more complex) models in the curvefit code
- What is controlling the range of the draws here? Unclear, and Indian data overall had a much smaller range than the IHME dashboard's US predictions' range.
- Try social distancing/intervention related covariates
  - weighted based on # interventions
  - weighted based on mobility change (FB colocation data)
  - weighted based on both; weigh an intervention according to the corresponding mobility change
    - i.e. schools being closed --> 0.2, night curfew --> 0.05
  - See [this notebook](../notebooks/ihme/mobility.ipynb) for a start
- IHME as a synthetic data generator: feed 1 week IHME predictions to SEIR, measure SEIR improvement/longevity. Also try the same with feeding SEIR into SEIR. See [this notebook](../notebooks/ihme/synth.ipynb) for a start

# Scripts

## [pipeline.py](../scripts/ihme/pipeline.py)

usage: `python3 pipeline.py -d mumbai -c config/mumbai.yaml>`
can also specify `-f path/to/output`

This runs the model [here](../models/ihme/model.py) on the params specified in `<config>.yaml` / `default.yaml`.

Outputs plots and csv in `outputs/ihme/forecast/`.

## [pipeline_all.py](../scripts/ihme/pipeline_all.py)

usage: `python3 pipeline_all.py -c config/mumbai.yaml -d mumbai`
can also specify `-f path/to/output`

This runs the model ` on the params specified in `<config>.yaml` / `default.yaml`, but ignores ycol and instead fits to all compartments

Outputs plots and csv in `outputs/ihme/forecast/`.

## [backtesting.py](../scripts/ihme/backtesting.py)

usage: `python3 backtesting.py -c config/mumbai.yaml -d mumbai`
can also specify `-f path/to/output`

This runs backtesting with the params specified in `<config>.yaml` / `default.yaml`, but ignores ycol and instead fits to all compartments. This model is rerun at the `increment` specified in the config, over all historical data, predicting `forecast_days` into the future.

Outputs plots and csv in `outputs/ihme/backtesting/`.

## [default.yaml](../scripts/ihme/config/default.yaml)

This is where the default config lives. Any other config files added in this directory will build off of the default one.

# main/ihme
## [main.py](../main/ihme/main.py)
Here lives the code to get the data, shape it, consume the model_parameters and config to set up appropriate arguments for the functions that run the model, which also live here.
Forecast and fitting are not separate exercises in this case, so there is no separation into a `forecast.py` here.

## [optmiser.py](../main/ihme/optmiser.py)
The hyperparameters in this case are the `fe_init` values, and support for n_days_train exists as well. Currently, there is a line in `optmiser.py` that forces the searchspace for `n_days` to be `(min_days, min_day+1)`, effectively seeting it to `min_days`. This was done to set it to be the minimum, as is done with the SEIR model.

## [backtesting.py](../main/ihme/backtesting.py)
This is where the backtester class lives, calling `IHMEBacktest.test` runs the backtesting.
This is powered under the hood by code from `fitting.py`

# models/ihme

## [model.py](../models/ihme/model.py)
This is the core model code. Uses `BasicModel` from `curvefit`.
- True fitting is currently done in `model.run()`, which is called in `model.predict()`. It takes a start/end date - evaluate if these dates should be different, and if so, move the call to `pipeline.run()` into `predict`.