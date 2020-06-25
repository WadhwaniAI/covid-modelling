# On Call Routine

Last Updated: June 10, Vishwa

## Typical Flow
1. Keshav/Puskar emails you with an ops request
2. Check how updated the data is on AthenaDB - if its more than a couple days outdated, ping Vasudha/Keshav/Puskar and find out if and when we'll be getting something newer.
3. Run `generate_report.py`. Details [below](#generate_reportpy).
4. This report contains everything you would need for a preliminary look: identify anything wrong, major alterations that might need to be made (param search spaces, data irregularities, etc)
5. Often there is a preliminary meeting and evaluation to determine any changes that may need to be made. However, as we move to more programatic model selection, this will likely phase out, so keep going...
6. Run `forecast.py`. (details [below](#forecastpy)) to generate forecasts at the deciles (+ a couple more percentiles requested). This file will generate csv's in the format Keshav + team can consume. You'll want to share the following:
   1. Everything in `what-ifs/`
   2. `deciles.csv`
   3. `deciles-params.csv`
7. Here onwards, anything more would be related to custom changes, implementing new changes in the code, etc.

## Scripts
see [scripts/oncall](https://github.com/WadhwaniAI/covid-modelling/tree/master/scripts/oncall)

For these, you'll want to also check `scripts/*config/` and see if there's anything there you need to modify. You can create a new one, change whatever is different from default, and run with that config. They all build on top of `default.yaml`.

### [generate_report.py](../scripts/oncall/generate_report.py)
Creates V1 of the report (see #4 above)-- none of the uncertainty estimation is in here; you'll see plots of M1 and M2 fits, M2 forecast based on M1, and top 10 trials/plots for those

This script calls `single_fitting_cycle`, which runs the model, and then takes its results and turns them into a report. Some light forecasting is included, but this is more to give us an idea of the performance than to be used to make a conclusive choice.

Command Line Parameters:

`-f --folder`: reports/assets will be stored in `reports/<folder>`; required

`-d --district`: district; required [mumbai or pune]

`-c --config`: path to config file; required [see `scripts/*/config`]

Things you may want to change IN the script
- You may want to try running with different parameters.
  - `which_compartments` just deceased and total_infected
  - maybe we'll want to play with the train_period?

### [forecast.py](../scripts/oncall/forecast.py)

Runs the improved uncertainty estimation Alpan pitched which we originally did on a Google Sheet. Creates the csvs Keshav + team will consume, and a plot `deciles.png` just for reference. Adds a few things to the report, so this can be reviewed all together in the first meeting if things (fits, params, etc) look good on the top. This script runs create_report again, and will recreate the report with additions. Pulls everything directly from the pkl.

Command Line Parameters:

`-f --folder`: where to find the pkl, addl assets will be saved in `reports/<folder>`; required

`-c --config`: path to config file; required [see `scripts/*/config`]


### [regenerate_report.py](../scripts/oncall/regenerate_report.py)
The purpose of this script was to solely recreate the report/assets from the pickle when functions were changed after the initial run.

Command Line Parameters:

`-f --folder`: where to find the pkl, addl assets will be saved in `reports/<folder>/timestamp/`; required

### [newcsvs.py](../scripts/oncall/newcsvs.py)
This script can be deleted after a week, keeping in case it's useful for iteration if forecast.py sees any issues. Was used to generate assets when we manually did the uncertainty estimation on a spreadsheet.

Command Line Parameters:

`-f --folder`: where to find the pkl, addl assets will be saved in `reports/<folder>/timestamp/`; required

`-d --district`: district name; required [mumbai, pune]

## Other relevant notes
- `create_report.py`: this is where the report creation function lives, turning a dict into markdown and eventually docx.
- `notebooks/seir` contains a bunch of notebooks that are/were related to this flow. You may find them useful to iterate on the scripts, but no guarantees that they are as consistently maintained.