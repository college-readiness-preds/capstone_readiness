# Compilation of functions to make life easier in the future


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import linregress
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def split_data(df, strat= None):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.16, random_state=123, stratify=df[strat])
    train, validate = train_test_split(train_validate,
                                       test_size=.2, 
                                       random_state=123, 
                                       stratify=train_validate[strat])
    return train, validate, test


#template for renaming columns
'''
    def rename_cols(df):
    df = df.rename(columns= {'column': 'new_name',
                             'column': 'new_name',
                             })
    
    return df
'''


#get a quick summary of a dataset to get familiar with it
def summarize(df):
    shape = df.shape
    info = df.info()
    describe = df.describe()
    distributions = df.hist(figsize=(24, 10), bins=20)
    #pairplot = sns.pairplot(df)
    return shape, info, describe, distributions, #pairplot




# this contains functions for plotting various charts with seaborn and matplotlib
#for quick analysis


#Bar Chart
def bar_chart(df, x_col, y_col, title):
    sns.barplot(data=df, x=x_col, y=y_col)
    plt.title(title)
    plt.show()



#Bar Chart with mean line
def mean_bar_plot(df, x, y, title='Bar Plot with Mean Line'):
    ax = sns.barplot(data = df, x=x, y=y)
    plt.title(title)
    
    # calculate the mean value
    mean = np.mean(df[y])
    
    # add a line for the mean value
    plt.axhline(mean, color='r', linestyle='dashed', linewidth=2)
    
    # add the mean value annotation 
    ax.text(0, mean + 0.01, 'Mean: {:.2f}'.format(mean), fontsize=12)
    plt.show()



#Scatter Plot
def scatter_plot(df, x, y, title):
    sns.scatterplot(data = df, x=x, y=y)
    plt.title(title)
    plt.show()


#Line Plot
def line_plot(df, x, y, title):
    sns.lineplot(data = df , x=x, y=y)
    plt.title(title)
    plt.show()


#Bar Chart with color
def bar_chart_with_color(df, x, y, color, title):
    sns.barplot(data = df , x=x, y=y, color= color)
    plt.title(title)
    plt.show()


#Crosstab
def crosstab_plot(df, x, y, normalize=False, title='Crosstab Plot'):
    ct = pd.crosstab(df[x], df[y])
    if normalize:
        ct = ct.div(ct.sum(1), axis=0)
    sns.heatmap(ct, annot=True, cmap='Blues')
    plt.title(title)
    plt.show()



#Pearson's R (Continuous 1 vs Continuous 2), tests for correlation #linear
def pearson_r(x, y):
    """
    Calculates the Pearson correlation coefficient (r) between two lists of data using scipy.stats.pearsonr.
    """
    r, p = pearsonr(x, y)
    return r



#Spearman (Continuous vs Continuous 2), tests for correlation #non-linear


def spearman_rho(x, y):
    """
    Calculates the Spearman rank correlation coefficient (rho) between two lists of data using scipy.stats.spearmanr.
    """
    rho, p = spearmanr(x, y)
    return rho





# One Sample T Test
def one_sample_ttest(target_sample, overall_mean, alpha = 0.05):
    t, p = stats.ttest_1samp(target_sample, overall_mean)
    
    return t, p/2, alpha


# T-Test One Tailed (Continuous 1 vs Discrete)

'''
compares mean of continuous variables for the different groups in 
the discrete variable
'''

def t_test_one_tailed(data1, data2, alpha=0.05, alternative='greater'):
    t, p = stats.ttest_ind(data1, data2, equal_var=False)
    if alternative == 'greater':
        p = p/2
    elif alternative == 'less':
        p = 1 - p/2
    if p < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return t, p


# T-Test Two Tailed (Continuous 1 vs Discrete)

'''
compares mean of continuous variables for the different groups in 
the discrete variable
'''

def t_test_two_tailed(data1, data2, alpha=0.05):
    t, p = stats.ttest_ind(data1, data2, equal_var=False)
    if p < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return t, p



# Chi-Square (Discrete vs Discrete)
# testing dependence/relationship of 2 discrete variables

def chi_square_test(data1, data2, alpha=0.05):
    chi2, p, dof, expected = stats.chi2_contingency(data1, data2)
    if p < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return chi2, p



#ANOVA (Continuous 1 vs Discrete)
def anova_test(data, groups, alpha=0.05):
    f_val, p_val = stats.f_oneway(*data)
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return f_val, p_val



# Sum of Squared Errors SSE
def sse(y_true, y_pred):
    sse = mean_squared_error(y_true, y_pred) * len(y_true)
    return sse


# Mean Squared Error MSE
def mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


#Root Mean Squared Error RMSE
def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# Explained Sum of Squares ESS
def ess(y_true, y_pred):
    mean_y = np.mean(y_true)
    ess = np.sum((y_pred - mean_y)**2)
    return ess


# Total Sum of Squares TSS
def total_sum_of_squares(arr):
    return np.sum(np.square(arr))


# R-Squared R2

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# Linear Regressions
'''
Quickly calculate r value, p value, and standard error
'''
def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err




# Explained Variance -or- RSquared
def explained_variance(y_true, y_pred):
    evs = explained_variance_score(y_true, y_pred)
    print("Explained Variance: ", evs)
    return evs




#isolating target variable in train, validate, test sets
def isolate_target(train, validate, test, target):

    X_Train = train.drop(columns = [target])
    y_Train = train[target]

    X_val = validate.drop(columns = [target])
    y_val = validate[target]

    X_test = test.drop(columns = [target])
    y_test = test[target]
    return X_Train, y_Train, X_val, y_val, X_test, y_test



def GLM(power, alpha):
    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_traintarget)

    # predict train
    y_train['value_pred_lm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = rmse(y_traintarget, y_train.value_pred_mean)

    # predict validate
    y_validate['value_pred_lm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = rmse(y_validatetarget, y_validate.value_pred_median)

    return print("RMSE for GLM using TweedieRegressor\nTraining/In-Sample: ", round(rmse_train), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate))


    

def DBSCAN_and_scatterplot(df, col1, col2, eps= .75, min_samples= 15, scaler_type= StandardScaler):
    #create new df with the columns given in the arguments
    df1 = df[[col1, col2]]
    #convert that new df into an array with float dtypes
    df1_array = df1.values.astype('float32', copy= False)
    #define scaler I want to use
    scaler = scaler_type().fit(df1_array)
    df1_array = scaler.transform(df1_array)
    #fit scaler_type to the array and assign to variable
    scaler = scaler_type().fit(df1_array)
    #reassign array variable to itself but transformed by the scaler
    df1_array = scaler.transform(df1_array)
    #perform dbscan on the new array with the parameters given in the arguments
    dbsc = DBSCAN(eps= eps, min_samples=min_samples).fit(df1_array)
    #define lables
    labels = dbsc.labels_
    #make new column for labels
    df['labels'] = labels
    #explore, show value_counts of our new labels, 
    value_counts = df.labels.value_counts()
    plot1 = sns.scatterplot(df[col1], df[col2], hue= df.labels)

    return df, value_counts, plot1