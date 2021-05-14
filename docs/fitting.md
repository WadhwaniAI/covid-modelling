We use black box optimisation algorithms to do the fitting. They are ideal for epi models as any form of gradient, or any other higher order optimisation method is not possible with them.

Specifically, we use the `hyperopt` package. It implements Sequential Model Based Optimisation (SMBO) via Tree Parzen Estimation (TPE). 

We have also implemented `gridsearch`, MCMC and Gaussian Process optimisation via `scikit-optimise`. 

The user can also implement another fitting method if they wish to. The fitting method must be a child of `main/seir/optimisers/base.py`.