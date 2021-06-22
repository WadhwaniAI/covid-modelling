# Wadhwani AI Covid Modelling

This repository holds the codebase for the Wadhwani AI Covid modelling
effort. During the epidemic (2020 and early 2021) this codebase was
developed and used by the Wadhwani AI team to provide estimates of
caseload and resource burden to various local governments around
India. This codebase is now freely available for others to use and to
build upon.

Our primary approach to forecasting covid-related outcomes was through
parameter estimation of SEIR-like models from count data (_confirmed,
active, recovered, deceased_) using hyper-parameter optimization
(hyperopt). However, the code has been abstracted such that that
initial case was an instance of a broader theme. There is support to
deal with a variety of data types and sources, forecasting models and
uncertainty, and techniques for parameter estimation. For example, in
addition to the SEIR-family, curve fit models like those developed at
IHME can be used; instead of using hyperopt, Bayesian parameter
estimation via Markov chain Monte Carlo (MCMC) is also supported. The
codebase design treats these concepts as compnents that can be swapped
in and out of a workflow depending on requirements or taste.

In addition to estimation and forecasting, this codebase includes
tools for visualizations; making it an end-to-end resource for public
health policy makers. There are several Jupyter Notebooks within this
repository that can be run to build a standard set of reports. More
advanced users can build custome report by piecing together other
components that this repository provides. Finally, developers can
extend the capabilities of this codebase by creating concrete modules
that extend our abstractions.

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