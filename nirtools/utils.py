import numpy as np
import random
# https://qiita.com/hmkz/items/0689cd85fb3e1adcda1a
def decode(a, b):
    """decode for GA-WLS.
    a: start index of the selected feature
    b: width of the selected region

    Examples)
    >>> decode(575, 10)
    array([575, 576, 577, 578, 579, 580, 581, 582, 583, 584])
    """
    return np.arange(a, a+b)

def truncate(a, a_min=0, a_max=None):
    """
    a: array
    a_min: minimum value in returned array
    a_max: maximum value in returned array 

    Examples)
    >>> truncate(decode(575, 10), 0, 580)
    array([575, 576, 577, 578, 579, 580])
    >>> truncate(decode(575, 10), 0)
    array([575, 576, 577, 578, 579, 580, 581, 582, 583, 584])
    """
    
    if a_max == None:
        return a[a_min<=a]
    else:
        a_min = min(a_min, min(a))
        return a[np.logical_and(a_min<=a, a<=a_max)]

def generate_start_width(n_features, width=10):
    """
    n_features, width
    >>> random.seed(66)
    >>> generate_start_width(650, width=50)
    [72, 20]
    >>> random.seed(42)
    >>> generate_start_width(650)
    [114, 1]
    """
    return [random.randint(0, n_features-1), random.randint(1, width)]

def translate(chromosome, n_features):
    """translate chromosome into indices for feature selection
    Examples)
    >>> translate( [[468, 1], [300, 4], [44, 4]], 650 )
    array([ 44,  45,  46,  47, 300, 301, 302, 303, 468])
    """
    indexes = np.concatenate(list(map(lambda ind:decode(*ind), chromosome)))
    indexes.sort()
    return truncate(indexes, 0, n_features-1)

def evaluate(individual, X, y, model, score_func, index_keyword=None):
    indexes = translate(individual, X.shape[1])
    if index_keyword is None:
        # basic model
        model.fit(X[:,indexes], y)
        return score_func(y, model.predict(X[:,indexes])),
    else:
        # complicated model
        model.set_params(**{index_keyword:indexes})
        model.fit(X, y)
        return score_func(y, model.predict(X)),
