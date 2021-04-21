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