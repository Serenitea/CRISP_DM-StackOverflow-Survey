import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

not_lang_list = ['Android',
 'AngularJS',
 'Arduino',
 'Arduino / Raspberry Pi',
 'Cassandra',
 'Cloud',
 'Cloud (AWS, GAE, Azure, etc.)',
 'Cordova',
 'Hadoop',
 'LAMP',
 'MongoDB',
 'Node.js',
 'ReactJS',
 'Redis',
 'SQL Server',
 'Salesforce',
 'SharePoint',
 'Spark',
 'Windows Phone',
 'Wordpress',
'WordPress', 'Write-In',
 'iOS']

lang_comm_list = ['Bash/Shell/PowerShell', 'C',
 'C#',
 'C++',
 'Clojure',
 'F#',
 'Go','HTML/CSS',
 'Java',
 'JavaScript',
 'Objective-C',
'Other(s)',
 'PHP',
 'Python',
 'R',
 'Ruby',
 'Rust',
 'SQL',
 'Scala',
 'Swift',
 'Visual Basic / VBA']

mixed_all_list = ['Android',
 'AngularJS',
 'Arduino',
 'Arduino / Raspberry Pi',
 'Assembly',
 'Bash/Shell',
 'Bash/Shell/PowerShell',
 'C',
 'C#',
 'C++',
 'C++11',
 'CSS',
 'Cassandra',
 'Clojure',
 'Cloud',
 'Cloud (AWS, GAE, Azure, etc.)',
 'Cobol',
 'CoffeeScript',
 'Common Lisp',
 'Cordova',
 'Dart',
 'Delphi/Object Pascal',
 'Elixir',
 'Erlang',
 'F#',
 'Go',
 'Groovy',
 'HTML',
 'HTML/CSS',
 'Hack',
 'Hadoop',
 'Haskell',
 'Java',
 'JavaScript',
 'Julia',
 'Kotlin',
 'LAMP',
 'Lua',
 'Matlab',
 'MongoDB',
 'Node.js',
 'Objective-C',
 'Ocaml',
 'Other(s):',
 'PHP',
 'Perl',
 'Python',
 'R',
 'ReactJS',
 'Redis',
 'Ruby',
 'Rust',
 'SQL',
 'SQL Server',
 'Salesforce',
 'Scala',
 'SharePoint',
 'Sharepoint',
 'Smalltalk',
 'Spark',
 'Swift',
 'TypeScript',
 'VB.NET',
 'VBA',
 'Visual Basic',
 'Visual Basic 6',
 'WebAssembly',
 'Windows Phone',
 'WordPress',
 'Wordpress',
 'Write-In',
 'iOS']

lang_uncomm_list = ['Assembly',
 'Cobol',
 'CoffeeScript',
 'Common Lisp',
 'Dart',
 'Delphi/Object Pascal',
 'Elixir',
 'Erlang',
 'Groovy',
 'Hack',
 'Haskell',
 'Julia',
 'Kotlin',
 'Lua',
 'Matlab',
 'Ocaml',
 'Perl',
 'Sharepoint',
 'Smalltalk',
 'TypeScript',
 'WebAssembly',]

lang_all_list = ['Assembly',
 'Bash/Shell',
 'Bash/Shell/PowerShell',
 'C',
 'C#',
 'C++',
 'C++11',
 'CSS',
 'Clojure',
 'Cobol',
 'CoffeeScript',
 'Common Lisp',
 'Dart',
 'Delphi/Object Pascal',
 'Elixir',
 'Erlang',
 'F#',
 'Go',
 'Groovy',
 'HTML',
 'HTML/CSS',
 'Hack',
 'Haskell',
 'Java',
 'JavaScript',
 'Julia',
 'Kotlin',
 'Lua',
 'Matlab',
 'Objective-C',
 'Ocaml',
 'Other(s):',
 'PHP',
 'Perl',
 'Python',
 'R',
 'Ruby',
 'Rust',
 'SQL',
 'Scala',
 'Sharepoint',
 'Smalltalk',
 'Swift',
 'TypeScript',
 'VB.NET',
 'VBA',
 'Visual Basic',
 'Visual Basic 6',
 'WebAssembly']

ord_feat = ['OpenSourcer', 'CareerSat', 'JobSat', 'MgrIdiot', 'MgrMoney', 
            'JobSeek', 'LastHireDate', 'WorkPlan', 'WorkRemote', 'ImpSyn', 'CodeRev', 
            'UnitTests', 'PurchaseHow', 'PurchaseWhat', 'SOPartFreq']
bool_feat = ['Hobbyist', 'BetterLife', 'ITperson', 'Trans', 'Dependents']
cat_feat = ['MainBranch', 'OpSys', 'Employment', 'Country', 'EdLevel', 
            'UndergradMajor', 'MgrWant', 'CompFreq', 'ResumeUpdate', 'WorkLoc', 'SocialMedia', 'Extraversion']
mixed_cat_feat = ['EduOther', 'DevType', 'LanguageWorkedWith', 
                   'DatabaseWorkedWith', 'PlatformWorkedWith',
                   'WebFrameWorkedWith', 'MiscTechWorkedWith',
                   'DevEnviron', 'Containers', 'JobFactors', 'WorkChallenge', 
                   'SONewContent', 'Gender', 'Ethnicity']
num_feat = ['ConvertedComp', 'OrgSize', 'YearsCode', 'Age1stCode', 'WorkWeekHrs', 'Age', 'CodeRevHrs']
feat_drop = ['CurrencySymbol', 'CompTotal', 'FizzBuzz', 'LastInt', 'BlockchainOrg', 
             'BlockchainIs', 'OffOn', 'ScreenName', 'SOVisit1st', 'SOTimeSaved', 'SOFindAnswer', 
             'SOHowMuchTime', 'SOAccount', 'SOJobs', 'SOVisitFreq', 'EntTeams', 'SOComm', 
             'WelcomeChange', 'SurveyLength', 'SurveyEase', 'OpenSource', 'Student', 'Sexuality', 
             'SOVisitTo', 'LanguageDesireNextYear', 'DatabaseDesireNextYear', 'PlatformDesireNextYear', 
            'WebFrameDesireNextYear', 'MiscTechDesireNextYear', 'CurrencyDesc']

#bool dict
bool_dict = {'Yes': 1, 'No': 0, 'SIGH': 1, 'Also Yes': 1, 'Fortunately, someone else has that title': 0}

#numerical data
OrgSize_dict = {'10,000 or more employees':10000, '5,000 to 9,999 employees':7500,
       '100 to 499 employees':300, '20 to 99 employees':60, '2-9 employees':6,
       '500 to 999 employees':750, '1,000 to 4,999 employees':3000,
       'Just me - I am a freelancer, sole proprietor, etc.':1,
       '10 to 19 employees':15}
YearsCode_dict = {'Less than 1 year': 0, 'Less than 1 year ': 0, 'More than 50 years': 50}
Age1stCode_dict = {'Younger than 5 years': 4, 'Older than 85': 86}

#dictionaries of categorical variables. genders simplified into Man, Woman, and non-binary
satis_dict = {'Very dissatisfied': 0, 
              'Slightly satisfied': 3,
              'Neither satisfied nor dissatisfied': 2,
              'Slightly dissatisfied': 1,
              'Very satisfied': 4}
OpenSourcer_dict = {'Never': 0, 'Less than once per year': 1, 
                    'Less than once a month but more than once per year': 2, 'Once a month or more often': 3}
MgrIdiot_dict = {'Not at all confident': 1, 'Very confident': 3,
       'Somewhat confident': 2, "I don't have a manager": 0}
MgrMoney_dict = {'Yes': 2, 'Not sure': 1, 'No': 0}
JobSeek_dict = {'I am actively looking for a job': 2, 
                'I’m not actively looking, but I am open to new opportunities': 1, 
                'I am not interested in new job opportunities': 0}
LastHireDate_dict = {"NA - I am an independent contractor or self employed": np.nan, 
                    "More than 4 years ago": 4, 
                     "3-4 years ago": 3, 
                     "1-2 years ago": 2,
                    "Less than a year ago": 1, 
                     "I've never had a job": 0}
WorkPlan_dict = {"There is a schedule and/or spec (made by me or by a colleague), and I follow it very closely": 2, 
                 "There is a schedule and/or spec (made by me or by a colleague), and my work somewhat aligns": 1, 
                 "There's no schedule or spec; I work on what seems most important or urgent": 0}
WorkRemote_dict = {"All or almost all the time (I'm full-time remote)": 5,
                   "More than half, but not all, the time": 4,
                   "About half the time": 3, 
                   "Less than half the time, but at least one day each week": 2,
                   "A few days each month": 1, 
                   "Less than once per month / Never": 0, 
                   "It's complicated": 0}
ImpSyn_dict = {"Far below average": 0, "Far above average": 4, "Average": 2, 
               "A little below average": 1, "A little above average": 3}
CodeRev_dict = {"Yes, because I was told to do so": 1, 
                "Yes, because I see value in code review": 2, "No": 0}
UnitTests_dict = {"Yes, it's part of our process": 3, 
                  "Yes, it's not part of our process but the developers do it on their own": 2, 
                  "No, but I think we should": 1, 
                  "No, and I'm glad we don't": 0}
PurchaseHow_dict = {"The CTO, CIO, or other management purchase new technology typically without the involvement of developers": 0,
                    "Not sure": 0, 
                    "Developers typically have the most influence on purchasing new technology": 2, 
                    "Developers and management have nearly equal input into purchasing new technology": 1}
PurchaseWhat_dict = {"I have some influence": 1, 
                     "I have little or no influence": 0, 
                     "I have a great deal of influence": 2}
SOPartFreq_dict = {"Multiple times per day": 4, 
                   "Less than once per month or monthly": 1, 
                   "I have never participated in Q&A on Stack Overflow": 0, 
                   "Daily or almost daily": 5, 
                   "A few times per week": 3,
                   "A few times per month or weekly": 2}
Student_dict = {'No': 0, 'Yes, full-time': 2, 'Yes, part-time': 1}

dict_dict = {"Hobbyist": bool_dict, 'BetterLife': bool_dict, 
             'ITperson': bool_dict, 'Trans': bool_dict, 'Dependents': bool_dict, 
            'OrgSize': OrgSize_dict, 'YearsCode': YearsCode_dict, 
             'YearsCodePro': YearsCode_dict,
             'OpenSourcer': OpenSourcer_dict, 'Age1stCode': Age1stCode_dict, 
             'CareerSat': satis_dict, 'JobSat': satis_dict, 
             'MgrIdiot': MgrIdiot_dict, 'MgrMoney': MgrMoney_dict, 
            'JobSeek': JobSeek_dict, 'LastHireDate': LastHireDate_dict, 
             'WorkPlan': WorkPlan_dict, 'WorkRemote': WorkRemote_dict, 
             'ImpSyn': ImpSyn_dict, 'CodeRev': CodeRev_dict, 
            'UnitTests': UnitTests_dict, 'PurchaseHow': PurchaseHow_dict, 
             'PurchaseWhat': PurchaseWhat_dict, 'SOPartFreq': SOPartFreq_dict}




dict_dict = {"Hobbyist": bool_dict, 'BetterLife': bool_dict, 
             'ITperson': bool_dict, 'Trans': bool_dict, 'Dependents': bool_dict, 
            'OrgSize': OrgSize_dict, 'YearsCode': YearsCode_dict, 
             'YearsCodePro': YearsCode_dict,
             'OpenSourcer': OpenSourcer_dict, 'Age1stCode': Age1stCode_dict, 
             'CareerSat': satis_dict, 'JobSat': satis_dict, 
             'MgrIdiot': MgrIdiot_dict, 'MgrMoney': MgrMoney_dict, 
            'JobSeek': JobSeek_dict, 'LastHireDate': LastHireDate_dict, 
             'WorkPlan': WorkPlan_dict, 'WorkRemote': WorkRemote_dict, 
             'ImpSyn': ImpSyn_dict, 'CodeRev': CodeRev_dict, 
            'UnitTests': UnitTests_dict, 'PurchaseHow': PurchaseHow_dict, 
             'PurchaseWhat': PurchaseWhat_dict, 'SOPartFreq': SOPartFreq_dict}
#Investigate missing data using different thresholds of %NaN missing.
def investigate_nan_threshold(df, interval, start):
    '''
    This function finds how many columns have more than a certain percentage 
    of data missing.

    INPUTS:
        start - the initial threshold percentage of data missing to be analyzed (e.g. 10%)
        interval - the amount of increase in the analysis threshold (e.g. 5%)
        if the previous threshold has at least 1 column remaining

    OUTPUTS:
        Prints the names of the columns that have more than the threshold % of 
        data missing as well as the current threshold.
    '''

    n = start
    df_rows, df_cols = df.shape
    missing_list = [1]
    while len(missing_list) > 0:
        missing_list = [col for col in df.columns if (df[col].isnull().sum()/df_rows)*100 > n]
        if len(missing_list) > 0:
            print('There are {} columns with more than {}% of data missing.'.format(len(missing_list), n))
            print(missing_list)
            print('--------------------------------------')
            n = n+interval
        else:
            return [col for col in df.columns if (df[col].isnull().sum()/df_rows)*100 > n-interval]
        
def assess_missing_col(df):
    '''
    This function analyzes how much data is missing from a dataframe and plots a histogram depicting the number of columns that have a certain percentage of missing data
    IN: dataframe to analyze NaN's for
    OUT: 
    - series with column names as the index and % of missing data as values
    - histogram plot of the series
    '''
    df_rows, df_cols = df.shape
    missing_num = pd.Series(df.isnull().sum(), name = 'Number of Missing')
    #all columns
    missing_per = pd.Series(missing_num/(df_rows)*100, name = '% NaN Missing')

    #only columns with missing data
    missing_data = pd.Series(missing_num[missing_num > 0]/df_rows*100, name = '% NaN Missing')
    missing_data.sort_values(inplace = True)
    print(missing_data)
    
    plt.hist(missing_data, bins = 50)
    plt.xlabel('Nan % in a column (%)')
    plt.ylabel('Counts')
    #plt.title('Histogram of missing value counts for each column')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', alpha=0.2)
    plt.show()

    return missing_data, plt.show()




def get_vals(df, col, delimiter):
    '''
    for years 2017-2019.
    processes a specified column from the raw imported dataframe
    for 2017, delimiter = '; '
    for 2018+19, delimiter = ';'
    '''
    df_count, df_vals = eval_complex_col(df, col)
    return df_vals

def process_col(df, col):
    '''
    for years 2016-2019.
    processes a a specified column from the raw imported dataframe
    '''
    s = df[col]
    s = s.dropna()
    s_len = s.shape[0]
    cat_count, col_vals = eval_complex_col(df, col)
    s_split = s.str.split(pat = '; ')
    return s,s_len, s_split, cat_count, col_vals


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
    cat_count = clean_series[0].value_counts()
    
    
#    print('Unique Categories: ', col_vals)
    return cat_count, col_vals


def clean_df_cat(df, to_remove):
    '''
    Removes columns that match any of the values in the to_remove list

    '''
    for item in to_remove:
        for col in df.columns:
            if item.casefold() == col.casefold():
                df = df.drop(col, axis = 1)
    return df

    '''
    2016-2019
    converts a series of lists into a df with each list as a row
    also returns a transposed version.
    '''

def s_of_lists_to_df(s):
    df = pd.DataFrame(item for item in s)
    df_transposed = df.transpose()
    return df, df_transposed



def make_df_bool(df, df_transposed, vals_list):
    '''
    for all years (2015-2019)
    creates a df of bool values based on whether each survey response has the value in vals_list.
    df: dataframe of survey responses, 
    vals_list: list of values for conditions of the new columns, with 1 col per val
    '''
    for item in vals_list:
        df[item] = df_transposed.iloc[:,:].isin([item]).sum()
    df_bool = df.loc[:,vals_list]
    return df_bool


#def process_data(df, col):
#    '''
#    Condensed function processing from initial imported df to boolean df
#    Used for 2017-2019
#    '''
#    df_droppedna, df_droppedna_len, df_droppedna_split, df_count, df_vals = process_col(df, col)
#    df2, df2_transposed = s_of_lists_to_df(df_droppedna_split)
#    df_bool = make_df_bool(df2, df2_transposed, df_vals)
    
#    return df_bool, df_vals, df_droppedna_len


def process_data_extended(df, col, delimiter):
    '''
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


def find_other_lang(df):
    '''
    for all years (2015-2019)
    must edit the df to match the lang_comm_list first!
    Finds the difference between the languages of the current year and the list of common languages
    Returns a list of columns to be merged into an "others" column
    '''
    other_lang = set(df.columns).difference(set(lang_comm_list))
    other_lang_list = list(other_lang)
    return other_lang_list

def make_counts2015(ini_df, series_name, index_name):
    '''
    for 2015 data - multiple columns for one data category
    converts a dataframe into a 1-d series to a 2 column df,
    returned df has Columns and column counts
    series sorted alphabetically
    '''
    series = ini_df.count()
    series = series.rename_axis(index_name)
    series = series.rename(series_name).sort_index()
    df = series.reset_index()
#    df.sort_values(by=[sort_by], inplace = True)
    return series, df



def make_counts(ini_df, series_name, index_name):
    '''
    2016-2019
    creates a df of counts for each column of the boolean df
    Series name: 'Counts'
    Index name: 'Language'
    '''
    series = ini_df.sum()
    series = series.rename_axis(index_name)
    series = series.rename(series_name).sort_index()
    df = series.reset_index()
#    df.sort_values(by=[sort_by], inplace = True)
    return series, df



def process_data_extended(df, col, delimiter):
    '''
    for years 2017-2019.
    processes a specified column from the raw imported dataframe
    for 2017, delimiter = '; '
    for 2018+19, delimiter = ';'
    IN: df, col, delimiter
    OUT:
    df_bool
    df_vals
    df_len
    '''
    s = df[col]
    s = s.dropna()
    df_len = s.shape[0]
    df_count, df_vals = eval_complex_col(df, col)
    s_split = s.str.split(pat = delimiter)

    df_new = pd.DataFrame(item for item in s_split)
    df_new_transposed = df_new.transpose()

    #
    for item in df_vals:
        df_new[item] = df_new_transposed.iloc[:,:].isin([item]).sum()
    df_bool = df_new.loc[:,df_vals]

    return df_bool, df_vals, df_len


def process_data(df):
    '''
    Input: original 2019 dataframe
    
    Steps:
    1. drop unwanted features
    2. drop rows that don't contain income
    3. drop rows that have more than 45% missing data
    4. convert the mixed variables to bool tables and merge them with the main table
    '''
    row_threshold = 45 #change row threshold here
    
    df = df.drop(feat_drop, axis = 1) #drop unwanted features columns
    #df = df[df['ConvertedComp'].isnull() == 0] #decided to not filter this early
    df_lowmiss = df[df.isnull().sum(axis=1) < row_threshold].reset_index(drop=True) #drop rows with over 45% missing data

    #remove rows that have more than 140hr worked per week
    df_filtered = df_lowmiss[df_lowmiss['WorkWeekHrs'] < 140]
    #return df_filtered


    for var in dict_dict:
        df_filtered[var].replace(dict_dict[var], inplace = True)

    #convert numerical features to floats
    #df_filtered[num_feat].astype(float)
    for var in num_feat:
        df_filtered[var] = pd.to_numeric(df_filtered[var])

    #make boolean dummies dfs out of mixed categorical features and merge table
    
    merged_df = df_filtered.copy()
    for var in mixed_cat_feat:
        var_bool, var_vals, var_len = process_data_extended(df_filtered, var, ';') #make new bool df with each value having its own column
        #print(var_vals)
        merged_df = pd.DataFrame.merge(merged_df, var_bool, how = 'outer',  left_index = True, right_index = True) #merge new bool df with main df
        merged_df = merged_df.drop(var, axis = 1) #drop the column that was just dummied

    #make dummies out of simple categorical features
    df_dummy = pd.get_dummies(merged_df[cat_feat].dropna().astype(str))
    df_wdummies = merged_df.join(df_dummy, )
    df_wdummies_dropped = df_wdummies.drop(cat_feat, axis = 1)
    
    return df_wdummies_dropped

<<<<<<< HEAD
def initial_preprocessing(df):
    '''
    Input: original 2019 dataframe
    
    Steps:
    1. drop unwanted features
    2. drop rows that don't contain income
    3. drop rows that have more than 45% missing data
    '''
    row_threshold = 45 #change row threshold here
    
    df = df.drop(feat_drop, axis = 1) #drop unwanted features columns
    #df = df[df['ConvertedComp'].isnull() == 0] #decided to not filter this early
    df_lowmiss = df[df.isnull().sum(axis=1) < row_threshold].reset_index(drop=True) #drop rows with over 45% missing data

    #remove rows that have more than 140hr worked per week
    df_filtered = df_lowmiss[df_lowmiss['WorkWeekHrs'] < 140]
    return df_filtered

def clean_lang_2019(bool_df):
    '''
    Conforms the code to match with a list of languages to be used for all 5 years.
    - Combines the 2 VB related columns together
    - Makes a list of languages to be classified as "others"
    - make a bool series of whether respondents use ANY of the "others" languages
    IN: df of bool of programming language usage by respondents
    OUT: series of bool of the usage of "other"/uncommon languages
    '''
    #rename cols to match the 5yr df
    bool_df = bool_df.rename(columns = {"VBA": "Visual Basic / VBA", "Other(s):": "Other(s)"})
    
    #make list of other langs
    other_lang2019_list = find_other_lang(bool_df)
    other_lang2019_list = sorted(other_lang2019_list)
    #combines the list of other langs into one col
    n = 0
    for elem in other_lang2019_list:
        bool_df["Other(s)"] = (bool_df["Other(s)"] | bool_df[elem])
        bool_df = bool_df.drop(elem, axis = 1) #drop the now unnecessary columns
    return bool_df

=======
>>>>>>> parent of 0a5b8e5... create proportion of language by dev types
