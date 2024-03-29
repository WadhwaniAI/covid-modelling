# Wadhwani AI Covid Modelling

**FEB 2023**: _We are no longer actively developing or supporting
this project; it has been
[archived](https://docs.github.com/en/repositories/archiving-a-github-repository/archiving-repositories).
Others who may want to use it for their own modelling or
experimentation are free to do so. For any questions,
please [email our team](mailto:covid-modellers@wadhwaniai.org)._

This repository holds the codebase for the Wadhwani AI Covid modelling
effort. During the epidemic (2020 and early 2021) this codebase was
developed and used by the Wadhwani AI team to provide estimates of
caseload and resource burden to various local governments around
India. This codebase is now freely available for others to use and to
build upon.

Our primary approach to forecasting covid-related outcomes was through
parameter estimation of compartmental, SEIR-like models from count data (_confirmed,
active, recovered, deceased_) using hyper-parameter optimization
(hyperopt). However, the code has been abstracted such that that
initial case was an instance of a broader theme. There is support to
deal with a variety of data types and sources, forecasting models and
uncertainty, and techniques for parameter estimation. For example, in
addition to the SEIR-family, curve fit models like those developed at
IHME can be used; instead of using hyperopt, Bayesian parameter
estimation via Markov chain Monte Carlo (MCMC) is also supported. The
codebase design treats these concepts as components that can be swapped
in and out of a workflow depending on requirements or taste.

In addition to estimation and forecasting, this codebase includes
tools for visualizations; making it an end-to-end resource for public
health policy makers. There are several Jupyter Notebooks within this
repository that can be run to build a standard set of reports. More
advanced users can build custom reports by piecing together other
components that this repository provides. Finally, developers can
extend the capabilities of this codebase by creating concrete modules
that extend our abstractions.

Please find the detailed documentation of this repo in this folder.

# Setting Up

### Clone the repo

`git clone --single-branch --branch master --depth 1 https://github.com/WadhwaniAI/covid-modelling.git`

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

- To read more about the codebase structure, [click here](docs/codebase_structure.md)
- To read more about the data, [click here](docs/data.md)
- To read more about data smoothing methods, [click here](docs/smoothing.md)
- To read more about SEIR models, [click here](docs/seir.md)
- To read more about IHME, [click here](docs/ihme.md)
- To read more about fitting process, [click here](docs/fitting.md)
- To read more about uncertainty estimation using ABMA, [click here](docs/abma.md)
- To read more about uncertainty estimation using MCMC, [click here](docs/mcmc.md)
- To read more about config file and how everything comes together, [click here](docs/config.md)
- To read more about the oncall process, [click here](docs/oncall.md)
