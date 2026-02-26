"""Test script for GAWLS

This demo fully relies on NIR Shootout 2002 data,
published by Eigenvector inc. http://eigenvector.com/data/tablets/nir_shootout_2002.mat
"""
from io import BytesIO
import requests
from pathlib import Path
import os
from scipy.io import loadmat
from itertools import product
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.metrics import r2_score
from nirtools.feature_selection import GAWLS

SHOOTOUT_DATA_ALIASES=[typ+"_"+vartyp
    for typ, vartyp in
    list(product(['calibrate','test','validate'],['1','2','Y']))
]
URL_SHOOTOUT2002 = "http://eigenvector.com/data/tablets/nir_shootout_2002.mat"

def load_shootout(dat, datatype="calibrate_1"):
    if datatype not in SHOOTOUT_DATA_ALIASES:
        raise ValueError("%s is not in %s"%(datatype, ", ".join(SHOOTOUT_DATA_ALIASES)))
    if datatype.endswith('Y'):
        return dat[datatype][0][0][5]
    else:
        return dat[datatype][0][0][5],dat[datatype][0][0][7][1][0][0]

def main():
    r = requests.get(URL_SHOOTOUT2002)
    dir_path =Path('./data')
    file_name = 'nir_shootout_2002.mat'
    file_path = dir_path / file_name
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        print("The folder exists, but proceed with the downloaded file.")
    with open(file_path, "wb") as mat_file:
        mat_file.write(BytesIO(r.content).read())
    data = loadmat(file_path)
    X_train, xaxis = load_shootout(data, 'calibrate_1')
    ytrain = load_shootout(data, 'calibrate_Y')
    X_test, _ = load_shootout(data, 'test_1')
    ytest = load_shootout(data, 'test_Y')

    y_train = ytrain[:,-1]/ytrain[:,0]
    y_test = ytest[:,-1]/ytest[:,0]
    base = GridSearchCV(PLSRegression(),
                        {'n_components':np.arange(1,11)})
    gawls_pls = GAWLS(base,
                    n_populations=10, n_generations=5)
    gawls_pls.fit(X_train, y_train)
    y_calc = gawls_pls.predict(X_train)
    print("R^2_train:", r2_score(y_train, y_calc))
    print("GAWLS best parameters chosen:",gawls_pls.best_params_)

if __name__=='__main__':
    main()