import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE


#----------------------------------------------------------------------------------------

# Function to plot residuals
def plot_residuals(y, yhat):
    
    sns.scatterplot(x=y, y=(yhat-y))
    plt.xlabel('Tax Value')
    plt.ylabel('Residuals')
    plt.show()

#---------------------------------------------------------------------------------------

# Calculates Regression Errors
def regression_errors(y, yhat):
    '''
    Takes in the target variable and the yhat(predictions) and calulates the SSE, MSE, RMSE, ESS, and TSS regression errors
    '''
    mse = mean_squared_error(y, yhat)
    rmse = mean_squared_error(y, yhat, squared=False)
    sse= mse * len(y)
    ess = yhat-y.mean()
    ess = ess ** 2
    ess = ess.sum()
    tss = ess + sse
    
    print(f'''
        Regression Errors
        -----------------
        SSE: {sse}
        MSE: {mse}
        RMSE: {rmse}
        ESS: {ess}
        TSS: {tss}
    ''')

    return sse, mse, rmse, ess, tss

#----------------------------------------------------------------------------------------

# Same function as above, but without print statement
def regression_errors2(y, yhat):
    '''
    Takes in the target variable and the yhat(predictions) and calulates the SSE, MSE, RMSE, ESS, and TSS regression errors
    '''
    mse = mean_squared_error(y, yhat)
    rmse = mean_squared_error(y, yhat, squared=False)
    sse= mse * len(y)
    ess = yhat-y.mean()
    ess = ess ** 2
    ess = ess.sum()
    tss = ess + sse

    return sse, mse, rmse, ess, tss
 
#----------------------------------------------------------------------------------------

# Baseline Mean Errors
def baseline_mean_errors(y):
    '''
    Takes in the target variable(y) and sets a baseline. Then calculates the SSE, MSE, and RMSE of the baseline.
    y = target
    '''
    baseline = np.repeat(y.mean(), len(y))
    
    mse= mean_squared_error(y, baseline)
    sse = mse * len(y)
    rmse = mean_squared_error(y, baseline, squared=False)

    print(f'''
        Baseline Mean Errors
        -----------------
        SSE: {sse}
        MSE: {mse}
        RMSE: {rmse}''')

    return sse, mse, rmse

#----------------------------------------------------------------------------------------

# Same Function as above but without print statment
def baseline_mean_errors2(y):
    '''
    Takes in the target variable(y) and sets a baseline. Then calculates the SSE, MSE, and RMSE of the baseline.
    y = target
    '''
    baseline = np.repeat(y.mean(), len(y))
    
    mse= mean_squared_error(y, baseline)
    sse = mse * len(y)
    rmse = mean_squared_error(y, baseline, squared=False)

    return sse, mse, rmse

#----------------------------------------------------------------------------------------

# Tells me if its better or worse than the baseline.
def better_than_baseline(y, yhat):
    '''
    Function that incorporates regression_error() and baseline_mean_error() functions in order to
    state if the model is better than the baseline.
    y = target
    yhat = predictions
    '''
    sse, mse, rmse, ess, tss = regression_errors(y, yhat)
    
    sse_base, mse_base, rmse_base = baseline_mean_errors(y)
    
    result = rmse_base - rmse

    if rmse_base < rmse:

        print('Results')
        print('------------------------------')
        print(f' RMSE Preds: {round(rmse,2)}')
        print(f' RMSE Baseline: {round(rmse_base,2)}')
        print(f' Difference: {round(result,2)}')
        print('Baseline is better')
    else:
        print('Results')
        print('------------------------------')
        print(f' RMSE Preds: {round(rmse,2)}')
        print(f' RMSE Baseline: {round(rmse_base,2)}')
        print(f' Difference: {round(result,2)}')
        print('MODEL IS BETTER!')

    return sse, mse, rmse, ess, tss, sse_base, mse_base, rmse_base

#----------------------------------------------------------------------------------------

# Select K Best Function
def select_kbest(x, y):
    
    '''
    Function that intakes X_train(predicitions) and y_train(target) and outputs the best 2 features using 
    SelectKBest model. Returns df of best features.
    '''
    
    # The trains
    X_train = x
    y_train = y
    
    # SelectKBest model object. Two best features being selected using f_regresssion.
    f = SelectKBest(f_regression, k=2)
    f.fit(X_train, y_train)
    
    # get_support() to retrieve kbest features
    f_mask = f.get_support()
    
    # iloc to get dataframe with kbest features
    k_best = X_train.iloc[:,f_mask]
    
    return k_best

#----------------------------------------------------------------------------------------

# Recursive Feature Elimination
def rfe(x, y):
    
    ''' Function that intakes the X_train and y_train in order to process it through
        the RFE model. This RFE model returns a dataframe of the top 2 features.
    '''
    
    # the trains
    X_train = x
    y_train = y
    
    # model that will be put into RFE model
    lm = LinearRegression()
    
    # RFE model and selecting best 2 features
    rfe = RFE(lm, n_features_to_select=2)
    
    # fitting RFE model
    rfe.fit(X_train, y_train)
    
    # retrieving best features
    ranks = rfe.ranking_
    
    # creating a list of column names
    columns = X_train.columns.tolist()
    
    # Creating df for the rankings
    feature_rankings = pd.DataFrame({'feature':columns, 'ranking':ranks})
    
    return feature_rankings.sort_values('ranking')

#----------------------------------------------------------------------------------------

# Linear Regression 
def linear(X_train, y_train, X_val, y_val, train, val):

    '''
    This function creates a Linear Rergression model for both X_train and X_val and returns their 
    regression and baseline errors. Then it prints out the Baseline RMSE, Train RMSE, and Validate RMSE.
    Then it gets the difference of Train RMSE and Validate RMSE.
    '''

    # Model Object and Fitted
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Predictions
    train['lm_preds'] = lm.predict(X_train)
    val['lm_preds'] = lm.predict(X_val)

    # Model Performance vs. Baseline
    tsse, tmse, trmse, tess, ttss, tsse_base, tmse_base, trmse_base = better_than_baseline2(y_train, train.lm_preds)
    vsse, vmse, vrmse, vess, vtss, vsse_base, vmse_base, vrmse_base = better_than_baseline2(y_val, val.lm_preds)

    result_train = trmse_base - trmse
    result_val = vrmse_base - vrmse
    difference = trmse - vrmse

    print('RMSE Results')
    print('-------------')
    print(f'Baseline: {round(trmse_base,2)}')
    print(f'Linear Regression Train Data: {round(trmse,2)}')
    print(f'Linear Regression Validate Data: {round(vrmse,2)}')
    print(f'Difference: {round(difference,2)}')
    
#----------------------------------------------------------------------------------------

def better_than_baseline2(y, yhat):
    '''
    Function that incorporates regression_error() and baseline_mean_error() functions in order to
    state if the model is better than the baseline.
    y = target
    yhat = predictions
    '''
    sse, mse, rmse, ess, tss = regression_errors2(y, yhat)
    
    sse_base, mse_base, rmse_base = baseline_mean_errors2(y)
    
    result = rmse_base - rmse

    return sse, mse, rmse, ess, tss, sse_base, mse_base, rmse_base

#----------------------------------------------------------------------------------------

def lassolars(X_train, y_train, X_val, y_val, train, val):

    '''This function creates a Lasso Lars model for both X_train and X_val and returns their 
    regression and baseline errors. Then it prints out the Baseline RMSE, Train RMSE, and Validate RMSE.
    Then it gets the difference of Train RMSE and Validate RMSE.
    '''

    # Model Object and Fitted
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train)

    # Predictions
    train['lars_preds'] = lars.predict(X_train)
    val['lars_preds'] = lars.predict(X_val)

     # Model Performance vs. Baseline
    tsse, tmse, trmse, tess, ttss, tsse_base, tmse_base, trmse_base = better_than_baseline2(y_train, train.lars_preds)
    vsse, vmse, vrmse, vess, vtss, vsse_base, vmse_base, vrmse_base = better_than_baseline2(y_val, val.lars_preds)

    result_train = trmse_base - trmse
    result_val = vrmse_base - vrmse
    difference = trmse - vrmse

    print('RMSE Results')
    print('-------------')
    print(f'Baseline: {round(trmse_base,2)}')
    print(f'Lasso Lars Train Data: {round(trmse,2)}')
    print(f'Lasso Lars Validate Data: {round(vrmse,2)}')
    print(f'Difference: {round(difference,2)}')

#----------------------------------------------------------------------------------------

def glm(X_train, y_train, X_val, y_val, train, val):

    '''This function creates Generalized Linear Model model for both X_train and X_val and returns their 
    regression and baseline errors. Then it prints out the Baseline RMSE, Train RMSE, and Validate RMSE.
    Then it gets the difference of Train RMSE and Validate RMSE.
    '''

    # Model Object and Fitted
    glm = TweedieRegressor(power=0, alpha =2)
    glm.fit(X_train, y_train)

    # Predictions
    train['glm_preds'] = glm.predict(X_train)
    val['glm_preds'] = glm.predict(X_val)

    # Model Performance vs. Baseline
    tsse, tmse, trmse, tess, ttss, tsse_base, tmse_base, trmse_base = better_than_baseline2(y_train, train.glm_preds)
    vsse, vmse, vrmse, vess, vtss, vsse_base, vmse_base, vrmse_base = better_than_baseline2(y_val, val.glm_preds)

    result_train = trmse_base - trmse
    result_val = vrmse_base - vrmse
    difference = trmse - vrmse

    print('RMSE Results')
    print('-------------')
    print(f'Baseline: {round(trmse_base,2)}')
    print(f'Generlized Linear Model Train Data: {round(trmse,2)}')
    print(f'Generalized Linear Model Validate Data: {round(vrmse,2)}')
    print(f'Difference: {round(difference,2)}')

#----------------------------------------------------------------------------------------

def polyreg(X_train, y_train, X_val, y_val, train, val):

    '''This function creates Polynomial Regression model for both X_train and X_val and returns their 
    regression and baseline errors. Then it prints out the Baseline RMSE, Train RMSE, and Validate RMSE.
    Then it gets the difference of Train RMSE and Validate RMSE.
    '''

    # Polynomial Feature 
    pf = PolynomialFeatures(degree=3)
    pf.fit(X_train, y_train)

    # Transforming X and y train using PolyFeatures
    X_train_d2 = pf.transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # Model Object and Fitted
    lm = LinearRegression(normalize=True)
    lm.fit(X_train_d2, y_train)

    # Predictions
    train['pf_preds'] = lm.predict(X_train_d2)
    val['pf_preds'] = lm.predict(X_val_d2)

    # Model Performance vs. Baseline
    tsse, tmse, trmse, tess, ttss, tsse_base, tmse_base, trmse_base = better_than_baseline2(y_train, train.pf_preds)
    vsse, vmse, vrmse, vess, vtss, vsse_base, vmse_base, vrmse_base = better_than_baseline2(y_val, val.pf_preds)

    result_train = trmse_base - trmse
    result_val = vrmse_base - vrmse
    difference = trmse - vrmse

    print('RMSE Results')
    print('-------------')
    print(f'Baseline: {round(trmse_base,2)}')
    print(f'Polynomial Regression Train Data: {round(trmse,2)}')
    print(f'Polynomial Regression Validate Data: {round(vrmse,2)}')
    print(f'Difference: {round(difference,2)}')

#----------------------------------------------------------------------------------------

def polytest(X_train, y_train, X_val, y_val, X_test, y_test, train, val, test):

    '''
    This functions creates a Polynomial Regression model for Validate and Test. Model is fitted to Train data set.
    Retrieves the baseline and regression errors from better_than_baseline2 function.
    Then prints out the baseline and RMSE results ofr Validate and Test. After, it calculates the difference
    between Validate and Test.
    '''

    # Polynomial Feature 
    pf = PolynomialFeatures(degree=3)
    pf.fit(X_train, y_train)

    # Transforming X and y train using PolyFeatures
    X_train_d3 = pf.transform(X_train)
    X_test_d3 = pf.transform(X_test)
    X_val_d3 = pf.transform(X_val)

    # Model Object and Fit
    lm = LinearRegression(normalize=True)
    lm.fit(X_train_d3, y_train)

    # Predictions
    val['pf_preds'] = lm.predict(X_val_d3)
    test['pf_preds'] = lm.predict(X_test_d3)

    # Model Performance vs. Baseline
    train_sse, train_mse, train_rmse, train_ess, train_tss, train_sse_base, train_mse_base, train_rmse_base =                         better_than_baseline2(y_train, train.pf_preds)
    vsse, vmse, vrmse, vess, vtss, vsse_base, vmse_base, vrmse_base = better_than_baseline2(y_val, val.pf_preds)
    tsse, tmse, trmse, tess, ttss, tsse_base, tmse_base, trmse_base = better_than_baseline2(y_test, test.pf_preds)

    
    result_train = trmse_base - trmse
    result_val = vrmse_base - vrmse
    difference = vrmse - trmse

    print('RMSE Results')
    print('-------------')
    print(f'Baseline: {round(train_rmse_base,2)}')
    print(f'Polynomial Regression Validate Data: {round(vrmse,2)}')
    print(f'Polynomial Regression Model Test Data: {round(trmse,2)}')
    print(f'Difference: {round(difference,2)}')



