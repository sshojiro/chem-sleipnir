"""Test script for GAWLS
"""
from scipy.io import loadmat
from shootout.data import loadshootout
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.metrics import r2_score
from nirtools.feature_selection import GAWLS

def main():
    data = loadmat('data/nir_shootout_2002.mat')
    X_train, xaxis = loadshootout(data, 'calibrate_1')
    ytrain = loadshootout(data, 'calibrate_Y')
    X_test, _ = loadshootout(data, 'test_1')
    ytest = loadshootout(data, 'test_Y')

    y_train = ytrain[:,-1]/ytrain[:,0]
    y_test = ytest[:,-1]/ytest[:,0]
    base = GridSearchCV(PLSRegression(),
                        {'n_components':np.arange(1,11)})
    gawls_pls = GAWLS(base,
                    n_populations=10, n_generations=5)
    gawls_pls.fit(X_train, y_train)
    y_calc = gawls_pls.predict(X_train)
    print(r2_score(y_train, y_calc))
    print(gawls_pls.best_params_)

if __name__=='__main__':
    main()