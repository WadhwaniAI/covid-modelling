# Folder Strucure

```
.
├── configs/
│   ├── seir/
│   ├── ihme/
│   ├── exper/
│   ├── simulated_data/
├── data/
│   ├── dataloader/
│   ├── processing/
├── docs/
│   ├── sphinx/
│   ├── *.md
├── main/
│   ├── ihme/
│   ├── seir/
│   ├───├─── optimisers/
│   ├───├─── uncertainty/
│   ├───├─── main.py/
├── models/
│   ├── ihme/
│   ├── seir/
├── notebooks/
│   ├── seir/
├── scripts/
│   ├── ihme/
│   ├── seir/
├── utils/
│   ├── fitting/
│   ├── generic/
├── viz/
```

## configs

Contains all configs. Refer to [config.md](config.md) for more details.

```
├── configs/
│   ├── seir/
│   ├── ihme/
│   ├── exper/
│   ├── simulated_data/
```
- `seir` : all SEIR configs
- `ihme` : all IHME configs
- `exper` : all experimentation configs. Generated N configs from 1 config, for experimentation
- `simulated_data` : all simulated data configs.

## data

Contains all dataloading and processing code. Refer to [data.md](data.md) for more details.

```
├── data/
│   ├── dataloader/
│   ├── processing/
```
- `dataloader` : contains all dataloaders. If the user wishes to add more dataloaders, they will be children of the `BaseLoader` class
- `processing` : contains code for processing all data post loading it. We wished to add a generic data transformations class and have all processing code be children of the base transformation class, but we were unable to do that.

## docs

Contains all documentation.

```
├── docs/
│   ├── sphinx/
│   ├── *.md
```

- `sphinx` : contains autodocs generated from docstrings of functions and classes
- `*.md` : Markdown documentation of different aspects of the project

## main

Contains the code for fitting the params of the model to data, and estimating uncertainty.

```
├── main/
│   ├── ihme/
│   ├── seir/
│   ├───├─── optimisers/
│   ├───├─── uncertainty/
│   ├───├─── main.py/
```

- `ihme` : functions for fitting IHME models
- `seir` : functions for fitting general models. Initially developed to fit only SEIR models, but then abstracted more with time to work with general models

## models

Contains the code for fitting the params of the model to data, and estimating uncertainty.

```
├── models/
│   ├── ihme/
│   ├── seir/
```

- `ihme` : the IHME model
- `seir` : All SEIR models. If the user wishes to create a new SEIR model, it must be a child of the base epi-model class

If the user wishes to add a new model class, they can create a folder for it, implement a base class for that model class, and then create models as children of that base class. 

## notebooks and scripts

Contains all runnable code. Most important is `notebooks/seir/[STABLE] generate_report.ipynb`.

```
├── notebooks/
│   ├── seir/
├── scripts/
│   ├── ihme/
│   ├── seir/
```

## utils

Fitting specific utils in `fitting`. Rest all in `generic`.

```
├── utils/
│   ├── fitting/
│   ├── generic/
```

## viz

All visualisation code

```
├── viz/
```

Additionally, 

- All local logging is typically done in `misc/reports`, but can be changed.
- All local data caching is done in `misc/cache`, but can be changed.
- Pyathena credentials are stored in `misc/pyathena`, but can be changed.

None of this is stored on git.