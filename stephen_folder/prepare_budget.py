import pandas as pd


##############################################
#...........For Budget Data Only.............#
##############################################


def prepare(df):

    '''Takes in full dataset cleans and returns only Per Student expenses with index set to spending category'''

    df = df.rename(columns= {'Unnamed: 0':'spending_category', '%':'pct'})
    df = df.drop(columns= ['AllFunds', '%.1', 'PerStudent.1'])
    df = df.iloc[7:38]
    df_temp = df.apply(lambda x: x.str.replace(',', ''))
    df_temp2 = df_temp.apply(lambda x: x.str.replace('$', ''))
    df = df_temp2.apply(lambda x: x.str.replace('%', ''))
    df = df[df['spending_category'] != 'Program\xa0expenditures\xa0by\xa0Program (Objects\xa06100-6400\xa0only)']
    df['GeneralFund'] = pd.to_numeric(df['GeneralFund'])
    df['pct'] = pd.to_numeric(df['pct'])
    df['PerStudent'] = pd.to_numeric(df['PerStudent'])
    df = df.drop(columns= ['GeneralFund', 'pct'])
    df.set_index('spending_category', inplace= True)

    return df





# split into two different dataframes, expend_by_function and expend_by_program

def function_program_split(df):

    ''' Takes in cleaned dataframe and splits into two dataframes, one for expenses by function 
        and one for expenses by program'''

    expend_by_function = df.iloc[1:14]
    expend_by_program = df.iloc[14:32]

    expend_by_function = expend_by_function.T
    expend_by_program = expend_by_program.T

    return expend_by_function, expend_by_program
