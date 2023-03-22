import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


#--------------------------------------------------------------------------------------------------


def extra_subs(train):

    '''    
    This function takes in train df and executes pearson r test testing for correlation 
    between each subject's passing rate and extracurricular spending per student.
    '''

    subjects = ['english_1', 'english_2', 'algebra', 'biology', 'history']

    data1 = []
    for i in subjects:

        r, p = stats.pearsonr(x=train.extracurricular_expend, y=train[i])
        data1.append({ 
                     'Correlation': r, 
                     'p-value': p
                     })
        
    df = pd.DataFrame(index=['English 1', 'English 2', 'Algebra', 'Biology', 'History'] ,data=data1)
    
    return df

#--------------------------------------------------------------------------------------------------

def spend_v_eco(school):
    '''
    this function takes in a dataframe and performs a independent t test between schools of high ecodis and low ecodis
    and returns t and p value
    '''
    low_ecodis = school[school['econdis'] < school.econdis.mean()].econdis
    high_ecodis = school[school['econdis'] > school.econdis.mean()].econdis
    t, p = ttest_ind(high_ecodis, low_ecodis, equal_var=False)
    return t, p

################################################################################

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

    
    ax = plt.gca()
    
    # Amount above bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate('{:.0f}'.format(height), (p.get_x()+p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=11)

    plt.xticks(X_axis, X)
    plt.xlabel("Subject")
    plt.ylabel("Percent Passing")
    plt.title("Percent of Students Passing STAAR Subjects Based on Teacher Experience")
    plt.ylim(60, 95)
    plt.grid(True, alpha=0.3, linestyle='--')
    leg = plt.legend(title="Teachers With 11+ Years of Experience")
    leg._legend_box.align = "left"
    plt.show()

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

# Question 1 Visual

def viz_abv_avg_staar(train):
    
    ma=abv_avg_staar_df(train)

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

    
    ax = plt.gca()
    
    # Amount above bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate('{:.0f}'.format(height), (p.get_x()+p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=11)

    plt.xticks(X_axis, X)
    plt.xlabel("Subject")
    plt.ylabel("Percent Passing")
    plt.title("STAAR Passing Rate for Economically Disadvantaged")
    plt.ylim(20, 95)
    plt.grid(True, alpha=0.3, linestyle='--')
    leg = plt.legend(title="Economically Disadvantaged")
    leg._legend_box.align = "left"
    plt.show()

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

# QUESTION 3 VISUAL

def viz_econdis_total_expend(train):
    
    ma = above_avg_econdis_total_expend(train)

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

    
    ax = plt.gca()
    
    # Amount above bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate('{:.0f}'.format(height), (p.get_x()+p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=9)

    plt.xticks(X_axis, X)
    plt.xlabel("Subject")
    plt.ylabel("Amount($)")
    plt.title("Total Expenditure for Economically Disadvantaged Schools")
    plt.ylim(2000, 14000)
    plt.grid(True, alpha=0.3, linestyle='--')
    leg = plt.legend(title="STAAR Passing Rate")
    leg._legend_box.align = "left"
    plt.show()

#--------------------------------------------------------------------------------------------------

#Question 6
def correlation_stu_teach_ratio_subject(train):

    '''    
    This function executes pearson r test checking for correlation 
    between each subject's passing rate and student teacher ratio
    and plots results.
    '''
    subjects = ['english_1', 'english_2', 'algebra', 'biology', 'history']

    data1 = []
    
    for i in subjects:

        r, p = stats.pearsonr(x=train.student_teacher_ratio, y=train[i])
        data1.append({
                     'Correlation': r, 
                     'p-value': p
                     })
        
    df = pd.DataFrame(index=['English 1', 'English 2', 'Algebra', 'Biology', 'History'] ,data=data1)
        #plot results
    graph = sns.relplot(data= df, x= df['Correlation'], y= df['p-value'], hue= df.index,s=200)
    graph.refline(y= 0.05, color= 'firebrick', linestyle= '-')
    plt.title('Correlation Significance and Strength')
    return df


#################################################################################################


def eco_experience(df):
    '''
    this function makes a table for the statistics tests and averages for teacher exp in economically disadvantaged schools
    '''
    train, validate, test= tts(df)
    train['teacher_0-10']=train['teacher_exp_0to5']+train['teacher_exp_6to10']
    a=train[(train['econdis']>58.5)&(train['teacher_exp_11_plus']>50)]
    b=train[(train['econdis']>58.5)&(train['teacher_0-10']>50)]
    
    subject=['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']
    
    more11plus=[round(a.english_1.mean(),2), round(a.english_2.mean(), 2), round(a.algebra.mean(),2),
                round(a.biology.mean(),2), round(a.history.mean(),2)]
    less11plus=[round(b.english_1.mean(),2), round(b.english_2.mean(),2), round(b.algebra.mean(),2),
                round(b.biology.mean(),2), round(b.history.mean(),2)]
    
    te1, pe1=stats.ttest_ind(a.english_1, b.english_1, alternative='greater')
    te2, pe2=stats.ttest_ind(a.english_2, b.english_2, alternative='greater')
    ta, pa=stats.ttest_ind(a.algebra, b.algebra, alternative='greater')
    tb, pb=stats.ttest_ind(a.biology, b.biology, alternative='greater')
    th, ph=stats.ttest_ind(a.history, b.history, alternative='greater')
    
    pval=[pe1, pe2, pa, pb, ph]
    
    data= pd.DataFrame(index=subject,data={
        'Passing Rate 0-10 Years': less11plus,
        'Passing Rate 11+ Years': more11plus,
        'p-value': pval
    }
                      )
    data=data.sort_values('p-value')
    
    return data

################################################################################################

def eco_ex_plot(df):
    '''
    this function plots the average passing rate of schools with majority of teachers experience
    '''
    ma=eco_experience(df)

    plt.figure(figsize=(10,5))
    X = ['English 1', 'English 2', 'Algebra', 'Biology', 'U.S. History']

    X_axis = np.arange(len(X))

    plt.bar(X_axis[0] - 0.1, ma['Passing Rate 0-10 Years'][0], 
            0.2, label = '0-10 Years Experience', color=['blue'], ec='black')
    plt.bar(X_axis[0] + 0.1, ma['Passing Rate 11+ Years'][0], 
            0.2, label = '11+ Years Experience', color=['orange'], ec='black')

    plt.bar(X_axis[1] - 0.1, ma['Passing Rate 0-10 Years'][1], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[1] + 0.1, ma['Passing Rate 11+ Years'][1], 0.2, color=['orange'], ec='black')

    plt.bar(X_axis[2] - 0.1, ma['Passing Rate 0-10 Years'][2], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[2] + 0.1, ma['Passing Rate 11+ Years'][2], 0.2, color=['orange'], ec='black')

    plt.bar(X_axis[3] - 0.1, ma['Passing Rate 0-10 Years'][3], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[3] + 0.1, ma['Passing Rate 11+ Years'][3], 0.2, color=['orange'], ec='black')

    plt.bar(X_axis[4] - 0.1, ma['Passing Rate 0-10 Years'][4], 0.2, color=['blue'], ec='black')
    plt.bar(X_axis[4] + 0.1, ma['Passing Rate 11+ Years'][4], 0.2, color=['orange'], ec='black')

    
    ax = plt.gca()
    
    # Amount above bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate('{:.0f}'.format(height), (p.get_x()+p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=11)
    
    plt.xticks(X_axis, X)
    plt.xlabel("Subject")
    plt.ylabel("Percent Passing")
    plt.title("Economically Disadvantaged Schools Passing STAAR Subjects Based on Majority Teacher Experience")
    plt.ylim(50, 95)
    plt.grid(True, alpha=0.3, linestyle='--')
    leg = plt.legend(title="Teacher Experience")
    leg._legend_box.align = "left"
    plt.show()

    #########################################################################################