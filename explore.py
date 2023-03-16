import pandas as pd
import numpy as np

#--------------------------------------------------------------------------------------------------

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

