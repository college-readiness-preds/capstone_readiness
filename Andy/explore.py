
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def tts(df, stratify=None):
    '''
    removing your test data from the data
    '''
    train_validate, test=train_test_split(df, 
                                 train_size=.8, 
                                 random_state=137,
                                 stratify=None)
    '''
    splitting the remaining data into the train and validate groups
    '''            
    train, validate =train_test_split(train_validate, 
                                      test_size=.3, 
                                      random_state=137,
                                      stratify=None)
    return train, validate, test

###########################################################################

def teacher_ex(df):
    '''
    This function will make a dataframe of the percent passing average staar score for schools with above and
    below average teachers with 11 years or more experience
    '''
    train, val, test = tts(df)
    subject=['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']
    
    low=train[train['teacher_exp_11_plus']<=48.5]
    high=train[train['teacher_exp_11_plus']>48.5]
    
    less11plus=[round(low.english_1.mean(),2), round(low.english_2.mean(), 2), round(low.algebra.mean(),2),
                round(low.biology.mean(),2), round(low.history.mean(),2)]
    more11plus=[round(high.english_1.mean(),2), round(high.english_2.mean(),2), round(high.algebra.mean(),2),
                round(high.biology.mean(),2), round(high.history.mean(),2)]
    
    te1, pe1=stats.ttest_ind(low.english_1, high.english_1, alternative='less')
    te2, pe2=stats.ttest_ind(low.english_2, high.english_2, alternative='less')
    ta, pa=stats.ttest_ind(low.algebra, high.algebra, alternative='less')
    tb, pb=stats.ttest_ind(low.biology, high.biology, alternative='less')
    th, ph=stats.ttest_ind(low.history, high.history, alternative='less')
    
    pval=[pe1, pe2, pa, pb, ph]
    
    data= pd.DataFrame(index=subject,data={
        'Above Average': more11plus,
        'Below Average': less11plus,
        'p-value': pval
    }
                      )
    return data

##############################################################################

def q2_plot(df):
    '''
    this function will plot the results for explore question 2
    '''
    ma=teacher_ex(df)

    plt.figure(figsize=(10,5))
    X = ['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']

    X_axis = np.arange(len(X))

    plt.bar(X_axis[0] - 0.1, ma['Above Average'][0], 0.2, label = 'Above Average', color=['blue'], ec='black')
    plt.bar(X_axis[0] + 0.1, ma['Below Average'][0], 0.2, label = 'Below Average', color=['orange'], ec='black')

    plt.bar(X_axis[1] - 0.1, ma['Above Average'][1], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[1] + 0.1, ma['Below Average'][1], 0.2, color=['orange'], ec='black')

    plt.bar(X_axis[2] - 0.1, ma['Above Average'][2], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[2] + 0.1, ma['Below Average'][2], 0.2, color=['orange'], ec='black')

    plt.bar(X_axis[3] - 0.1, ma['Above Average'][3], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[3] + 0.1, ma['Below Average'][3], 0.2, color=['orange'], ec='black')

    plt.bar(X_axis[4] - 0.1, ma['Above Average'][4], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[4] + 0.1, ma['Below Average'][4], 0.2, color=['orange'], ec='black')


    plt.xticks(X_axis, X)
    plt.xlabel("Subject")
    plt.ylabel("Percent Passing")
    plt.title("Percent of Students Passing STAAR Subjects Based on Teacher Experience")
    plt.ylim(60, 95)
    plt.grid(True, alpha=0.3, linestyle='--')
    leg = plt.legend(title="Teachers With 11+ Years of Experience")
    leg._legend_box.align = "left"
    plt.show()