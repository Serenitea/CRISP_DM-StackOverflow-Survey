import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

def poly_reg(X_var, y_var, degree):
    '''
    Performs polynomial regression
    '''
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X_var, 
                                                        y_var, 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    poly_fit = np.poly1d(np.polyfit(X_train, y_train, degree))
    from sklearn.metrics import r2_score
    return(r2_score(y_test, poly_fit(X_test)))

def calc_reg_score(X_var, y_var, reg_type, degree, score_type):
    '''
    Will split and train a polynomial regression model and returns either the R sqaured score or the accuracy score.
    IN:
        X_var -  Pandas dataframe of features
        y_var - Pandas series of variable to be predicted
        reg_type - 'linear' or 'polynomial'
        degree - degree of polynomial model
        score_type allowed - 'r2_score', 'accuracy_score'
    OUT: the R sqaured score or the accuracy score of the trained model.
        
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_var, 
                                                        y_var, 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    y_train = np.array(y_train).reshape(-1, 1)
    if reg_type == 'polynomial':
        poly_fit = np.poly1d(np.polyfit(X_train, y_train, degree)) #train model
        
        if score_type == 'r2_score':
            #from sklearn.metrics import r2_score
            return(r2_score(y_test, poly_fit(X_test)))

        elif score_type == 'accuracy_score':
            #from sklearn.metrics import accuracy_score
            return(accuracy_score(y_test, poly_fit(X_test)))

    elif reg_type == 'linear':
        from sklearn.linear_model import LinearRegression
        lm_model = LinearRegression(normalize=True) # Instantiate
        lm_model.fit(X_train, y_train) #train model

        if score_type == 'r2_score':
            #from sklearn.metrics import r2_score
            return(r2_score(y_test, lm_model(X_test)))

        elif score_type == 'accuracy_score':
            #from sklearn.metrics import accuracy_score
            return(accuracy_score(y_test, lm_model(X_test)))

def cal_lm_score(X_var, y_var, score_type):
    '''
    Will split and train a linear regression model and returns either the R sqaured score or the accuracy score.
    IN: 
        X_var -  Pandas dataframe
        y_var - Pandas series
        score_type allowed - r2_score, accuracy_score
    OUT: the R sqaured score or the accuracy score.
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split 
    X = X_var
    y = y_var
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train)


    if score_type == 'r2_score':
        from sklearn.metrics import r2_score
        return(r2_score(y_test, lm_model(X_test)))

    elif score_type == 'accuracy_score':
        from sklearn.metrics import accuracy_score
        return(accuracy_score(y_test, lm_model(X_test)))

def calc_score(df, score_type):
    '''
    trains a linear regression model and returns R score
    '''
    dfgrid = pd.DataFrame(columns=df.columns, index=df.columns)
    for col in dfgrid:
        for n_rows in range(len(dfgrid[col])):
            row = dfgrid.index.values[n_rows]
            #print(dfgrid.loc[row, col])
            dfgrid.loc[row, col] = lin_reg(df[col], df[row])
    return dfgrid

def calc_accuracy(df):
    '''
    trains a linear regression model and returns R score
    '''
    dfgrid = pd.DataFrame(columns=df.columns, index=df.columns)
    for col in dfgrid:
        for n_rows in range(len(dfgrid[col])):
            row = dfgrid.index.values[n_rows]
            #print(dfgrid.loc[row, col])
            dfgrid.loc[row, col] = lin_reg(df[col], df[row])
    return dfgrid

def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test
