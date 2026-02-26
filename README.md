A Light Library to support Chemometrics!
===

Genetic algorithm (GA) is a widely used heuristic optimization algorithm. GA-Wavelength selection, known as GAWLS, is a heuristic feature selection method to support chemometrics, especially for modelling near infrared (NIR) spectra.

## Feature of this library

The library is based on scikit-learn coding practice and pipeline approach, which allows users to combine the feature selection library with any of machine learning models.

The GA base library is [DEAP](https://deap.readthedocs.io/en/master/). The intention of this library is to facilitate coding chemometrics with GA.

## Usage

You can start using the feature selection library, integrating it with scikit-learn models such as PLS.

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from nirtools.feature_selection import GAWLS
base = GridSearchCV(PLSRegression(),
                    {'n_components':np.arange(1,11)})
gawls_pls = GAWLS(base,
                n_populations=10, n_generations=5)
gawls_pls.fit(X_train, y_train)
y_calc = gawls_pls.predict(X_train)
print("R^2_train:", r2_score(y_train, y_calc))
print("GAWLS best parameters chosen:",gawls_pls.best_params_)
```

## Getting started

You can start with executing the test script as follows.
One of the famous datasets in Chemometrics, called Shootout 2002, is used for the demonstration.

```bash
$ python tests/test_gawls_pls.py
The folder exists, but proceed with the downloaded file.
100%|##########| 5/5 [00:02<00:00,  1.97it/s]
root:INFO [2026-02-27 00:06:22,282]| Population: [[[343, 1], [373, 8], [385, 5]], [[343, 1], [373, 8], [385, 5]], [[480, 6], [373, 8], [385, 5]], [[480, 6], [38, 3], [385, 5]], [[480, 6], [373, 8], [385, 5]], [[343, 1], [373, 8], [385, 5]], [[480, 6], [38, 3], [385, 5]], [[480, 6], [373, 8], [385, 5]], [[343, 1], [373, 8], [385, 5]], [[480, 6], [38, 3], [385, 5]]]
root:INFO [2026-02-27 00:06:22,282]| Best population: [[480, 6], [373, 8], [385, 5]]
R^2_train: 0.9709208259834365
GAWLS best parameters chosen: [373 374 375 376 377 378 379 380 385 386 387 388 389 480 481 482 483 484
 485]
```