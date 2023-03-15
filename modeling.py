import pandas as pd
import numpy as np


#--------------------------------------------------------------------------------------------------

def the_trains(train,validate,test, target='target'):
    
    '''
    Creates X and y trains.
    '''
    # X and y trains sets
    X_train = train.drop([target], axis=1)
    y_train = train[target]

    # X and y validates sets
    X_validate = validate.drop([target], axis=1)
    y_validate = validate[target]

    # X and y test sets
    X_test = test.drop([target], axis=1)
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#--------------------------------------------------------------------------------------------------