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

    diff = results['Validate Baseline RMSE'] - results['Validate Model RMSE']
    reduction = diff / results['Validate Baseline RMSE']
    pct_improvement = round(reduction*100 ,2)
    pct_improvement

    results['Validate Improvement'] = pct_improvement
    results['Validate Improvement'] = results['Validate Improvement'].astype(str) + '%'


    return results

################################################################################################


def modeling_visual(df):
    '''
    this function shows the visual for modeling results
    '''
    ma=model_results(df)

    plt.figure(figsize=(10,5))
    X = ['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']

    X_axis = np.arange(len(X))

    plt.bar(X_axis[0] - 0.3, ma['Train Baseline RMSE'][0], 0.2, 
            label = 'Baseline RMSE', color=['blue'], ec='black')
    plt.bar(X_axis[0] - 0.1, ma['Train Model RMSE'][0], 0.2, 
            label = 'Train RMSE', color=['orange'], ec='black')
    plt.bar(X_axis[0] + 0.1, ma['Validate Model RMSE'][0], 0.2, 
            label = 'Validate RMSE', color=['rebeccapurple'], ec='black')

    plt.bar(X_axis[1] - 0.3, ma['Train Baseline RMSE'][1], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[1] - 0.1, ma['Train Model RMSE'][1], 0.2, color=['orange'], ec='black')
    plt.bar(X_axis[1] + 0.1, ma['Validate Model RMSE'][0], 0.2, color=['rebeccapurple'], ec='black')
    
    plt.bar(X_axis[2] - 0.3, ma['Train Baseline RMSE'][2], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[2] - 0.1, ma['Train Model RMSE'][2], 0.2, color=['orange'], ec='black')
    plt.bar(X_axis[2] + 0.1, ma['Validate Model RMSE'][0], 0.2, color=['rebeccapurple'], ec='black')

    plt.bar(X_axis[3] - 0.3, ma['Train Baseline RMSE'][3], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[3] - 0.1, ma['Train Model RMSE'][3], 0.2, color=['orange'], ec='black')
    plt.bar(X_axis[3] + 0.1, ma['Validate Model RMSE'][0], 0.2, color=['rebeccapurple'], ec='black')

    plt.bar(X_axis[4] - 0.3, ma['Train Baseline RMSE'][4], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[4] - 0.1, ma['Train Model RMSE'][4], 0.2, color=['orange'], ec='black')
    plt.bar(X_axis[4] + 0.1, ma['Validate Model RMSE'][4], 0.2, color=['rebeccapurple'], ec='black')


    plt.xticks(X_axis, X)
    plt.xlabel("Subject")
    plt.ylabel("RMSE")
    plt.title("Modeling Results")
    plt.ylim(0, 25)
    plt.grid(True, alpha=0.3, linestyle='--')
    leg = plt.legend(title="RMSE")
    leg._legend_box.align = "left"
    plt.show()