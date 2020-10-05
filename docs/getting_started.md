Hi!

The focus here in this project is to model covid19 cases in different geographical regions, and accurately forecast future cases.

# Setting Up

### Clone the repo

`git clone https://github.com/WadhwaniAI/covid-modelling.git`

If you prefer using SSH, use SSH

### Install packages

`pip install -r requirements.txt`

It would be highly recommended that you use either conda or virtualenv

### Run notebook

run the cells in `notebooks/seir/[STABLE] generate_report.ipynb` ([here](../notebooks/seir/)) end to end to get an idea of the end-to-end modelling pipeline.

# Further Reading

- To read more about the data, [click here](data.md)
- To read more about data processing and smoothing, [click here](smoothing.md)
- To read more about SEIR models, [click here](seir.md)
- To read more about IHME, [click here](ihme.md)
- To read more about fitting process, [click here](fitting.md)
- To read more about uncertainty estimation, [click here](uncertainty.md)
- To read more about config file and how everything comes together, [click here](config.md)
- To read more about the oncall process, [click here](oncall.md)
- If you want to contribute, do check out [general principles](general_principles.md) before writing code