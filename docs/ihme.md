# Model Reference Links

[IMHE CurveFit Documentation](https://ihmeuw-msca.github.io/CurveFit/)

[IMHE CurveFit Code Repo](https://github.com/ihmeuw-msca/CurveFit)

[IMHE paper](https://www.medrxiv.org/content/10.1101/2020.03.27.20043752v1.full.pdf)

[waiting for] IMHE technical writeup

[IMHE dashboard - US/Europe](https://covid19.healthdata.org/united-states-of-america)

# Code Implementation

## run_pipeline.py

usage: `python3 run_pipeline.py -p <key from params.json>`
try `python3 run_pipeline.py -p india_all`

This runs `BasicModel` on the params specified in `params.json`.

Outputs plots and a csv in `output/pipeline/<params_key>`.

**Future explorations required:**
- BasicModel versus the other defined (more complex) models in the curvefit code
- What is controlling the range of the draws here? Unclear, and Indian data overall had a much smaller range than the IHME dashboard's US predictions' range.

## data.py

Houses methods that read and clean up data to be in the following format, ready for consumption by run_pipeline.py. Column Names must be specified accordingly in `params.json`. Recommended names are:
- `date`: datetime
- `day`: int
- `group`: str or int
- `cases`: cumulative cases
- `deaths`: cumulative deaths

## params.json

Top level keys are names passed into the command line argument for `run_pipeline`. The rest are as follows:
- `data_func`: corresponding function in `data.py` that will return the dataframe,
- `data_func_args`: any args that must be passed into `data_func`,
- `test_size`: number of rows to withhold as a test set from the end of the data,
- `alpha_true`: true alpha,
- `beta_true`: true beta,
- `p_true`: true p,
- `xcol`: name of the _day_ columne,
- `date`: name of the column with the datetime date,
- `groupcol`: name of the column used for grouping,
- `ycols`: dictionary where keys are names of the columns to use as `y`, such as cases or deaths, and values are string names of functions from `curvefit.utils.functions` such as “erf”, “derf”, “log_erf”, etc.
- `daysforward`: number of days forward to predict,
- `daysback`: number of days before the start of the data to predict,
- `n_draws`: number of draws,
- `cv_threshold`: 1e-2,
- `smoothed_radius`: [2,2],
- `num_smooths`: 3,
- `exclude_groups`: [],
- `exclude_below`: 0,
- `exp_smoothing`: null,
- `max_last`: null

See `params.json["template"]` for an example.