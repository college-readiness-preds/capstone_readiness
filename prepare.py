import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------

def convert_dollars_to_float(x):
    fix = []
    blah = []
    something = [] #this list holds only the balance keys from the dictionary
    only_dollars = [] # this holds the values without the commas
    convert = []
    for d in x:
        d = d.replace(' ', '')
        fix.append(d)
    for e in fix:
        e = e.replace('-', '')
        blah.append(e)
        # this holds the converted values 
    for a in blah:
        a = a.strip('$')
        something.append(a)
    for b in something:
        b = b.replace(",","")
        only_dollars.append(b)
    for c in only_dollars:
        c = float(c)
        convert.append(c)

    return convert

#--------------------------------------------------------------------------------------------------

def change_dollars(df):
    df = df[df['salary'] != '-']
    df = df[df['salary'] != '?']
    df.salary = convert_dollars_to_float(df.salary)
    df.all_fund = convert_dollars_to_float(df.all_fund)
    df.extra_fund = convert_dollars_to_float(df.extra_fund)
    return df

#--------------------------------------------------------------------------------------------------

def rename_cols(df):
    df = df.rename(columns= {'eng1': 'english_1',
                         'eng2': 'english_2',
                         'ebel': 'bilingual_or_english_learner',
                         'ex_5' : 'teacher_exp_5',
                         'ex_10': 'teacher_exp_6to10',
                         'ex_1120': 'teacher_exp_11to20',
                         'ex_2130': 'teacher_exp_21tp30',
                         'ex_plus' : 'teacher_exp_over30',
                         'extra_fund': 'extracurricular_expend',
                         'all_fund': 'total_expend',
                         'ratio': 'student_teacher_ratio'
                         })
    return df

#--------------------------------------------------------------------------------------------------

def remove_symbols(df):
    rows = df.columns.to_list()
    for row in rows:
        df = df[df[row] != '-         ']           
        df = df[df[row] != '*         ']
    return df

#--------------------------------------------------------------------------------------------------

def clean_df():
    
    '''
    Cleans dataframe by replacing special characters with empty space
    and converting all columns to a numerical data type.
    '''
    
    # Load CSV
    df = pd.read_csv('school_data.csv', index_col=0)

    # Reset Index
    df = df.reset_index().drop('index', axis=1)

    # Covert dollar signs and special characters
    df = change_dollars(df)
    
    # Remove '*' and '-'
    df = remove_symbols(df)
    
    # Rename Columns
    df = rename_cols(df)
    
    # Columns to loop
    loop_columns = list(df.columns[1:])
    
    # Loop to remover '%' sign and change to float data type
    for col in loop_columns:
        try:
            df[col] = df[col].str.replace('%','').astype(float)
            
        except:
            continue
            
    df=combine_features(df)
    return df

#--------------------------------------------------------------------------------------------------

def combine_features(df):
    '''
    this function will combine features to targets and drop originals
    '''
    df['teacher_exp_0to5']=df['beginning_teach']+df['teacher_exp_5']
    df['teacher_exp_11_plus']= df['teacher_exp_11to20']+df['teacher_exp_21tp30']+df['teacher_exp_over30']
    df['high_edu']=df['masters']+df['doct']
    df=df.drop(columns=['masters', 'doct', 'beginning_teach', 'teacher_exp_5', 'teacher_exp_11to20', 
                    'teacher_exp_21tp30', 'teacher_exp_over30'])
    return df

def extra_v_sub(df):
    plt.xticks(rotation=90)
    plt.scatter(x = df.extracurricular_expend, y = df.algebra)
    plt.scatter(x = df.extracurricular_expend, y = df.english_1)
    plt.scatter(x = df.extracurricular_expend, y = df.english_2)
    plt.scatter(x = df.extracurricular_expend, y = df.biology)
    plt.scatter(x = df.extracurricular_expend, y = df.history)
    classes = ['algebra','english_1','english_2', 'biology', 'history']
    plt.legend(labels=classes)
    plt.title('Subject vs Expense for Extracurricular')
    plt.xlabel('Expense per Student on Extracurricular Activities')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, linestyle='--')
    return plt.show()



def expense_per_stu(df):
    
    
    # Low/High Economically Disadvantaged
    high = df[df['econdis'] > df['econdis'].mean()]
    low = df[df['econdis'] <= df['econdis'].mean()]
    
    # Plot histograms
    plt.hist(high.total_expend, bins=65, ec='black', color='blue')
    plt.hist(low.total_expend, bins=65, ec='black', color='orange')
    plt.xlim([0,35000])
    classes = ['High Economically Disadvantaged', 'Low Economically Disadvantaged']
    plt.legend(labels=classes, title='Economically Disadvantaged')
    plt.title('Total Expediture for High/Low Economically Disadvantaged Schools')
    plt.xlabel('Expense(per student)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.show()


