import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

#doesn't work for counting unique values, replaced by get_vals
def eval_complex_col(df, col):
    '''
    IN:
    df[col] - every str consists of one or more values (e.g. 'a, b, d')

    OUT:
    col_vals - All unique elements found in the column, listed alphabetically

    '''
    col_num = df[df[col].isnull() == 0].shape[0]
    col_df = df[col].value_counts().reset_index()
    col_df.rename(columns={'index': col, col:'count'}, inplace = True)
    col_series = pd.Series(col_df[col].unique()).dropna()
    clean_list = col_series.str.split(pat = ';').tolist()


    flat_list = []
    for sublist in clean_list:
        for item in sublist:
            flat_list.append(item)
    clean_series = pd.DataFrame(flat_list)
    clean_series[0] = clean_series[0].str.strip()

    col_vals = clean_series[0].unique()
    col_vals = pd.Series(sorted(col_vals))
    #cat_count = clean_series[0].value_counts() #this only counts values that are by themselves (the series is only made up of that value)


#    print('Unique Categories: ', col_vals)
#    return cat_count, col_vals
return col_vals

def process_col(df, col):
    '''
    for years 2016-2019.
    processes a a specified column from the raw imported dataframe
    '''
    s = df[col]
    s = s.dropna()
    s_len = s.shape[0]
    col_vals = get_vals(df, col)
    s_split = s.str.split(pat = '; ')
    return s,s_len, s_split, cat_count, col_vals

def process_col_extended(df, col, delimiter):
    '''
    previously process_col_extended
    for years 2017-2019.
    processes a specified column from the raw imported dataframe

    IN:
        df: SO Developer survey as dataframe
        col: column to be evaluated
        delimiter:
            for 2017, delimiter = '; '
        for 2018+19, delimiter = ';'
    OUT:
        df_bool: boolean dataframe of each extracted feature
        df_vals: list of all features extracted
        df_len: number of rows
    '''
    s = df[col]
    s = s.dropna()
    df_len = s.shape[0]
    df_count, df_vals = eval_complex_col(df, col)
    s_split = s.str.split(pat = delimiter)

    df_new = pd.DataFrame(item for item in s_split)
    df_new_transposed = df_new.transpose()

    for item in df_vals:
        df_new[item] = df_new_transposed.iloc[:,:].isin([item]).sum()
    df_bool = df_new.loc[:,df_vals]

    return df_bool, df_vals, df_len

def total_count(df, col1, col2):
    '''
    INPUT:
    df - the pandas dataframe you want to search
    col1 - the column name you want to look through
    col2 - the column you want to count values from
    look_for - a list of strings you want to search for in each row of df[col]

    OUTPUT:
    new_df - a dataframe of each look_for with the count of how often it shows up
    '''
    look_for = get_vals(df, col1, ';')
    from collections import defaultdict
    new_df = defaultdict(int)
    #loop through list of ed types
    for val in look_for:
        #loop through rows
        for idx in range(df.shape[0]):
            #if the ed type is in the row add 1
            if val in df.loc[idx, col1]:
                new_df[val] += int(df[col2][idx])
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1, col2]
    new_df.sort_values('count', ascending=False, inplace=True)
    return new_df

#possible_vals = ["Take online courses", "Buy books and work through the exercises",
                 "None of these", "Part-time/evening courses", "Return to college",
                 "Contribute to open source", "Conferences/meet-ups", "Bootcamp",
                 "Get a job as a QA tester", "Participate in online coding competitions",
                 "Master's degree", "Participate in hackathons", "Other"]

def clean_and_plot(df, title='Method of Educating Suggested', plot=True):
    '''
    INPUT
        df - a dataframe holding the CousinEducation column
        title - string the title of your plot
        axis - axis object
        plot - bool providing whether or not you want a plot back

    OUTPUT
        study_df - a dataframe with the count of how many individuals
        Displays a plot of pretty things related to the CousinEducation column.
    '''
    possible_vals =
    study = df['CousinEducation'].value_counts().reset_index()
    study.rename(columns={'index': 'method', 'CousinEducation': 'count'}, inplace=True)
    study_df = total_count(study, 'method', 'count', possible_vals)

    study_df.set_index('method', inplace=True)
    if plot:
        (study_df/study_df.sum()).plot(kind='bar', legend=None);
        plt.title(title);
        plt.show()
    props_study_df = study_df/study_df.sum()
    return props_study_df

props_df = clean_and_plot(df)


def col_of_lists_to_bool(df, col, delimiter = ';'):
    '''
    (previously named process_col_extended)
    for years 2017-2019.
    processes a specified column from the raw imported dataframe,
    where every cell is a list stored as a string

    IN:
        df: SO Developer survey as dataframe
        col: name of column to be evaluated
        delimiter:
            for 2016 and 2017, delimiter = '; '
        for 2018 and 2019, delimiter = ';'
        sort_order: choose to sort series outputted by ascending or descending order\
            True: ascending
            False: descending
        create_plot: create a plot instead of returning series of counts and proportions
        plot_type: type of plot to be created. options listed below
            ‘bar’ or ‘barh’ for bar plots
            ‘hist’ for histogram
            ‘box’ for boxplot
            ‘kde’ or ‘density’ for density plots
            ‘area’ for area plots
            ‘scatter’ for scatter plots
            ‘hexbin’ for hexagonal bin plots
            ‘pie’ for pie plots
    OUT:
        df_bool: list of all features extracted
        df_len: number of rows
        s_counts: series of counts of all values
        s_prop: series of proportion of all values

    '''
    s = df[col] #retrieve specified column
    s = s.dropna() #drop empty rows
    df_len = s.shape[0] #find number of remaining rows in the column
    df_vals = get_vals(df, col, delimiter) #retrieve all unique values in the column
    s_split = s.str.split(pat = delimiter) #split each string into lists

    df_new = pd.DataFrame(item for item in s_split) #explode series into dataframe, each list in the series turns into one item per column
    df_new_transposed = df_new.transpose() #transpose the dataframe for ease of counting each list

    for item in df_vals:
        df_new[item] = df_new_transposed.iloc[:,:].isin([item]).sum()
    df_bool = df_new.loc[:,df_vals] #make a bool dataframe with a column for each unique value
    return df_bool, df_vals, df_len

def process_data(df_filtered):
    '''
    Input: 2019 dataframe filtered by the initial_preprocessing function
    convert the mixed variables to bool tables and merge them with the main table
    '''

    for var in dict_dict:
        df_filtered[var].replace(dict_dict[var], inplace = True)

    #convert numerical features to floats
    #df_filtered[num_feat].astype(float)
    for var in num_feat:
        df_filtered[var] = pd.to_numeric(df_filtered[var])

    #make boolean dummies dfs out of mixed categorical features and merge table

    merged_df = df_filtered.copy()
    for var in mixed_cat_feat:
        var_bool, var_vals, var_len = process_col_of_lists(df_filtered, var, ';') #make new bool df with each value having its own column
        #print(var_vals)
        merged_df = pd.DataFrame.merge(merged_df, var_bool, how = 'outer',  left_index = True, right_index = True) #merge new bool df with main df
        merged_df = merged_df.drop(var, axis = 1) #drop the column that was just dummied

    #make dummies out of simple categorical features
    df_dummy = pd.get_dummies(merged_df[cat_feat].dropna().astype(str))
    df_wdummies = merged_df.join(df_dummy, )
    df_wdummies_dropped = df_wdummies.drop(cat_feat, axis = 1)

    return df_wdummies_dropped
