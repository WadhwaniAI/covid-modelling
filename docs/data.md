# Data

Irrespective of the data source, the fundamental requirement for all modelling is have to time series of aggregate counts. Those aggregate counts can correspond to a district, state, country, or any geographical entity one wishes to model for. 
Specifically, for a particular geographical location, for a specific date, we want a 4-tuple of (total, active, recovered, deceased) cases.
- total : total cases up until that date
- recovered : total recoveries up until that date
- deceased : total deaths up until that date
- active : (total - recovered - deceased) on that date

Therefore there are 3 independent time series here. 

A few important pointers here :
- We currently train on the cumulative numbers rather than daily incidences. There is a lot of noise in the reporting of daily incidences which led to this modelling decision. However, that may change in the future.
- For training, all compartments (total active, recovered, and deceased) are not required, a subset works, but it is highly recommended that they all are provided
- Data for some dates in the middle can be missing, but it highly recommended that the time series be as continuous as possible.

Typically, it is expected that every data source will at least provide the above time series data. Some data sources may provide more data, and that is talked about in detail for each data source.

# Data Sources

Currently, we have implemented dataloaders from 4 data sources :
- Covid19India
- Rootnet
- Athena
- JHU

## Covid19India

The wonderful set of volunteers at [covid19india](https://covid19india.org/) have been diligently maintaining the site for a long time now. Among a lot of other things, they have time series of aggregate counts for every district, state in India from the end of April. They have an [API](https://api.covid19india.org/) for accessing the data. However, the exact API for getting aggregate counts has changed with time, so we have multiple implementations - 

- [`raw_data`](https://api.covid19india.org/raw_data14.json) (patient level data; not implemented/code deprecated as API underwent too much flux)
- [`districts_daily`](https://api.covid19india.org/districts_daily.json) (was default for a long time but covid19india dropped support)
- [`data_all`](https://api.covid19india.org/v4/data-all.json) (current default)

Further, `data_all` also had time series data for tests, and migrations (if any), for every district and state. 

Covid19india also provides us with population projections for every state based on a 2019 Ministry of Statistics report. We aren't using it as of now.

## Rootnet

Currently not used as covid19india provides us with historical data for all states. It was implemented at a time when getting historical data for states from covid19india was hard/not possible. 
The API for accessing their data is [here](https://api.rootnet.in/). Their state level data is sourced from MoHFW website. One of the members of the volunteer group, NirantK has worked/works with them.

## Athena

This was implemented to setup a proper pipeline for accessing custom data from different goverment organisations that we serve. Someone from the government organisation/our organisation would input the data in a google sheet with a particular defined schema. That google sheet is then linked to AWS Athena, which allows us to query that google sheet like an SQL database, without creating a backend database. 

The user can check out [this](https://wadhwaniai.github.io/covid-data/) link to get a better idea of how to get credentials to access this data, tables supported by Athena and their respective schemas. 

Currently this data source is used only for Mumbai data (our primary customer). It was earlier used for Pune as well. `new_covid_case_summary` is the table that is used. 

Mumbai further provides us with "stratified data" : data where the active column is stratified further on different basis. It can be stratified in 3 different ways:

- severity (of infection)
    - [`asymptomatic`, `symptomatic`, `critical`]
- bed (type of bed on which patient is recovering)
    - [`ventilator`, `icu`, `o2_beds`, `non_o2_beds`, `hq`]
- facility (type of facility where the patient of recovering)
    - [`icu`, `dchc`, `dch`, `ccc2`]

Legend : 
- `o2_beds` : Oxygen Support
- `non_o2_beds` : No Oxygen Support
- `hq` : Home Quarantine
- `icu` : Intensive Care Unit
- `dchc` : Dedicated Covid Health Center
- `dch` : Dedicated Covid Hospital
- `ccc2` : Covid Care Center 2

In all methods of stratification, the sum total of the stratified columns equals to active for every date. 

Currently we use only stratification by bed type and severity as that is what our customer, BMC has shown interest in (Also, different facilities now have people of different severity levels and bed types, so different facilities not characteristically distinct). 

## JHU

This gets data from the [JHU dashboard](https://github.com/CSSEGISandData/COVID-19/) which tracks time series of cases for all countries and specific subregions of certain countries. The main use case of this data is for running experiments of different models across different geographies. 

# Code

The classes for loading data from the above sources have been implemented [here](../data/dataloader/). There are 4 classes - `Covid19IndiaLoader, RootnetLoader, AthenaLoader, JHULoader`. Each class has a `_load_data` function that returns a dict of `pd.DataFrame`s. Filtering/Processing of that data is done [here](../data/processing/processing.py). The function user is going to interact with most is `get_data` in `data/processing/processing.py`.

If the user wishes to implement a new class of dataloader, that class should be a child of the abstract class in `data/dataloader/base.py` (as are all other classes).