Hi!

This codebase is a framework for fitting blackbox models like epi-models (eg SEIR) using black box optimisation techniques (using hyperopt), and estimating uncertainty on top of optimisation trials, using an approximate technique we developed called ABMA.

It was initially developed to deal with covid casecount data (ie, Confirmed, Active, Recovered, Deceased numbers), SEIR models, and optimisation using hyperopt. However, the code has been abstracted to deal with different data types/sources (ie, any multivariate timeseries data source), any blackbox model, and any blackbox optimisation method. 

Please find the detailed documentation of this repo in this folder.

# Setting Up

### Clone the repo

`git clone https://github.com/WadhwaniAI/covid-modelling.git`

If you prefer using SSH, use SSH

### Install packages

`pip install -r requirements.txt`

It would be highly recommended that you use either conda or virtualenv. We have a very long `requirements.txt`, so some packages there may be old and unavailable.

If any line fails, install the packages using - 

`cat requirements.txt | xargs -n 1 pip install`

### Test everything out

run the cells in `notebooks/seir/[STABLE] generate_report.ipynb` ([here](../notebooks/seir/)) end to end to get an idea of the end-to-end modelling pipeline.

# Docstrings

Detailed function and class level documentation can be found in [sphinx](sphinx). The HTML files are not on git, but the instructions to create them are given in [sphinx/README.md](sphinx/README.md)

# Further Reading

- To read more about the codebase structure, [click here](codebase_structure.md)
- To read more about the data, [click here](data.md)
- To read more about data smoothing methods, [click here](smoothing.md)
- To read more about SEIR models, [click here](seir.md)
- To read more about IHME, [click here](ihme.md)
- To read more about fitting process, [click here](fitting.md)
- To read more about uncertainty estimation using ABMA, [click here](abma.md)
- To read more about uncertainty estimation using MCMC, [click here](mcmc.md)
- To read more about config file and how everything comes together, [click here](config.md)
- To read more about the oncall process, [click here](oncall.md)
- If you want to contribute, do check out [general principles](general_principles.md) before writing code