import numpy as np
import pandas as pd
import prepare as pr
import matplotlib.pyplot as plt
import seaborn as sns
import explore as ex

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def model_results(df):
    '''
    this function will calculate baseline RMSE for train/validate and generate the linear regression models/preds/RMSE and show the difference
    '''
    train, val, test= ex.tts(df)
    target=['english_1', 'english_2', 'algebra','biology', 'history']
    subject=['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']
    rmse_base_t=[]
    rmse_base_v=[]
    rmse_train=[]
    rmse_val=[]
    
    X_train=train.drop(columns=['biology', 'english_1', 'english_2', 'algebra', 'history', 'school_id'])
    X_val=val.drop(columns=['biology', 'english_1', 'english_2', 'algebra', 'history', 'school_id'])
    X_test=test.drop(columns=['biology', 'english_1', 'english_2', 'algebra', 'history', 'school_id'])
    
    lm=LinearRegression()
    for t in target:
        y_train=pd.DataFrame(train[t])
        y_val=pd.DataFrame(val[t])
        y_test=pd.DataFrame(test[t])
        
        lm.fit(X_train, y_train[t])
        pred_t= lm.predict(X_train)
        pred_v= lm.predict(X_val)
        
        train['Baseline Mean'] = train[t].mean()
        rmse_base_t.append(mean_squared_error(train[t], train['Baseline Mean'], squared=False))
        val['Baseline Mean'] = val[t].mean()
        rmse_base_v.append(mean_squared_error(val[t], val['Baseline Mean'], squared=False))
        
        rmse_train.append(mean_squared_error(y_train[t], pred_t, squared=False))
        rmse_val.append(mean_squared_error(y_val[t], pred_v, squared=False))
        
    results=pd.DataFrame(index=subject, data= {
        'Train Baseline RMSE': rmse_base_t,
        'Validate Baseline RMSE': rmse_base_v,
        'Train Model RMSE': rmse_train,
        'Validate Model RMSE': rmse_val,
    })
    results['Model Difference']=results['Train Model RMSE']- results['Validate Model RMSE']
    return results