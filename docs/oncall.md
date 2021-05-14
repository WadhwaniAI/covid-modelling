# What is on call

For forecasting caseload for our clients, instead of having 1 person do the forecasting, we had a monthly rolling system, where different RFs/interns were responsible for the forecasts to be sent out to our clients. This encapsulates the workflow for the oncall

# On Call Routine

Last Updated: May 14th 2021, Sansiddh

## Typical Flow
1. The product manager (Keshav/Puskar/Anshika) emails you with an ops request
2. Check how updated the data is on AthenaDB - if its more than a couple days outdated, ping Vasudha/Keshav/Puskar and find out if and when we'll be getting something newer.
3. Run `[STABLE] generate_report.ipynb`. Details [below](#generate_report).
4. This report contains everything you would need for a preliminary look: identify anything wrong, major alterations that might need to be made (param search spaces, data irregularities, etc)
5. Often there is a preliminary meeting and evaluation to determine any changes that may need to be made. However, as we move to more programatic model selection, this will likely phase out, so keep going..
7. Here onwards, anything more would be related to custom changes, implementing new changes in the code, etc.

## Scripts
see [scripts/oncall](https://github.com/WadhwaniAI/covid-modelling/tree/master/scripts/oncall)

For these, you'll want to also check `scripts/*config/` and see if there's anything there you need to modify. You can create a new one, change whatever is different from default, and run with that config. They all build on top of `default.yaml`.

### [generate_report](../notebooks/seir/[STABLE] generate_report.ipynb)
Creates V1 of the report (see #4 above)-- you'll see plots of the fits, best fit/forecasts, uncertainty forecasts/params, and top 10 trials/plots for those, decile plots, and ensemble mean forecast plots

This script calls `single_fitting_cycle`, which runs the model, and then takes its results and turns them into a report. 

All of the fitting is done via the input config file, which is given at the start of the notebook.

The last cells of the notebook - `create_output` and `create_report` create the csvs Keshav + team will consume, and a plot `deciles.png` just for reference. Adds a few things to the report, so this can be reviewed all together in the first meeting if things (fits, params, etc) look good on the top. This script runs create_report again, and will recreate the report with additions. Pulls everything directly from the pkl.

## Other relevant notes
- `create_report.py`: this is where the report creation function lives, turning a dict into markdown and eventually docx.
- `notebooks/seir` contains a bunch of other notebooks that are/were related to this flow. You may find them useful to iterate on the scripts, but no guarantees that they are as consistently maintained.