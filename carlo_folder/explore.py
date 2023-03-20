
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#--------------------------------------------------------------------------------------------

def split_data(df):
    ''' 
        This function is the train, validate, test, function.
        1. First we create the TRAIN and TEST dataframes at an 0.80 train_size( or test_size 0.2).
        2. Second, use the newly created TRAIN dataframe and split that at a 0.70 train_size
        ( or test_size 0.3), which means 70% of the train dataframe, so 56% of all the data.
        Now we have a train, validate, and test dataframes
    '''
    
    # Split the stuff
    train, test = train_test_split(df, train_size=0.8, random_state=137)
    train, validate = train_test_split(train, train_size = 0.7, random_state = 137)
    
    return train, validate, test

#--------------------------------------------------------------------------------------------

def abv_avg_staar_df(train):
    
    '''
    Subsets train dataset between schools that are above average economically disadvantaged and
    schools that have less economoically disadvantaged students. 
    
    Then it calculated the average STAAR passing rate between the two subsets and inputs it into a dataframe. 
    
    Two-sample T-Test is used to very statistical significane and is included in dataframe.
    '''
    
    # Subjects
    subject=['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']
    
    # above and below average Economically Disadvantaged
    low = train[train['econdis'] <= train.econdis.mean()]
    high = train[train['econdis'] > train.econdis.mean()]
    
    low_avg_staar = [round(low.english_1.mean(),2), round(low.english_2.mean(), 2), round(low.algebra.mean(),2),
                round(low.biology.mean(),2), round(low.history.mean(),2)]
    more_avg_staar = [round(high.english_1.mean(),2), round(high.english_2.mean(),2), round(high.algebra.mean(),2),
                round(high.biology.mean(),2), round(high.history.mean(),2)]
    
    te1, pe1 = stats.ttest_ind(low.english_1, high.english_1)
    te2, pe2 = stats.ttest_ind(low.english_2, high.english_2)
    ta, pa = stats.ttest_ind(low.algebra, high.algebra)
    tb, pb = stats.ttest_ind(low.biology, high.biology)
    th, ph = stats.ttest_ind(low.history, high.history)
    
    pval = [pe1, pe2, pa, pb, ph]
    
    data = pd.DataFrame(index=subject,data={
        'Above Average': more_avg_staar,
        'Below Average': low_avg_staar,
        'p-value': pval
    }
                      )
    return data

#--------------------------------------------------------------------------------------------

def above_avg_econdis_total_expend(train, subject='subject'):

    '''
    This function takes a Pandas DataFrame "train" and an optional argument "subject", and performs a 
    statistical analysis to compare the average total expenditure per student for two groups of schools:

    Above average economically disadvantaged schools with higher than average STAAR passing rates.
    Above average economically disadvantaged schools with lower than average STAAR passing rates.
    The function first selects the subset of schools with an economically disadvantaged percentage 
    above or equal to the mean. 

    It then splits this subset into two groups based on the "subject" column. The average total 
    expenditure per student is computed for each group, and a two-sample t-test is performed to 
    determine whether the difference in means is statistically significant.
    '''

    #  above average Economically Disadvantaged
    econdis_above_avg = train[train['econdis'] >= train.econdis.mean()]
    
    # Above and Below avg STAAR passing rate
    above_avg_staar = econdis_above_avg[econdis_above_avg[subject] > econdis_above_avg[subject].mean()]
    below_avg_staar = econdis_above_avg[econdis_above_avg[subject] <= econdis_above_avg[subject].mean()]
    
    # Total expenditure for above_avg_staar and below_average_staar
    avg_expend_above = above_avg_staar['total_expend'].mean()
    avg_expend_below = below_avg_staar['total_expend'].mean()
    
    # Above and Below total expenditures for stats test
    above = above_avg_staar['total_expend']
    below = below_avg_staar['total_expend']
    
    # Stats 2-Sample T-Test
    t, p = stats.ttest_ind(below, above)
    
    print('Expenditure')
    print('-----------')
    print(f'''The average expediture per student for students at above average economically 
            disadvantaged schools with higher average STAAR passing rates is 
            {avg_expend_above}.''')
    print()
    print(f'''The average expediture per student for students at above average economically 
            disadvantaged schools with lower than average STAAR passing rates is 
            {avg_expend_below}.''')
    print()
    print(f'''P-value: {p}''')
    
    return avg_expend_above, avg_expend_below, p

#--------------------------------------------------------------------------------------------------
