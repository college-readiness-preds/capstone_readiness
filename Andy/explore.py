
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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