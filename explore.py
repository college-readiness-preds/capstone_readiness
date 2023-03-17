import pandas as pd
import numpy as np



#--------------------------------------------------------------------------------------------------

# QUESTION 1

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

#--------------------------------------------------------------------------------------------------

# QUESTION 3

def above_avg_econdis_total_expend(train):
    
    # Above average Economically Disadvantaged
    econdis_above_avg = train[train['econdis'] >= train['econdis'].mean()]
    
    above = []
    below = []
    p_val = []
    
    subject_list = ['algebra','english_1','english_2','biology','history']
    
    for s in subject_list:
    
        # Above and Below avg STAAR passing rate
        above_avg_staar = econdis_above_avg[econdis_above_avg[s] > econdis_above_avg[s].mean()]
        below_avg_staar = econdis_above_avg[econdis_above_avg[s] <= econdis_above_avg[s].mean()]
        
        # Total expenditure for above_avg_staar and below_average_staar
        avg_expend_above = above_avg_staar['total_expend'].mean()
        avg_expend_below = below_avg_staar['total_expend'].mean()
    
        # Above and Below total expenditures for stats test
        above_stats = above_avg_staar['total_expend']
        below_stats = below_avg_staar['total_expend']
    
        # Stats 2-Sample T-Test
        t, p = stats.ttest_ind(below_stats, above_stats)
    
        above.append(avg_expend_above)
        below.append(avg_expend_below)
        p_val.append(p)
        
        
    
    # Subjects for plot
    subjects = ['Algebra', 'English 1', 'English 2', 'Biology', 'History']
    
    df = pd.DataFrame(index=subjects, data={
        'Above Average': above,
        'Below Average': below,
        'p-value': p_val})
  

    return df

#--------------------------------------------------------------------------------------------------

