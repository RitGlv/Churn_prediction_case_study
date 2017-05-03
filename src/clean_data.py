import numpy as np
import pandas as pd

def create_prediction(df):
    # converting to DatetimeIndex
    df['date_last_trip'] = pd.DatetimeIndex(df['last_trip_date'])
    # df['date_signup'] = pd.DatetimeIndex(df['signup_date'])

    # Engineer the churn column
    today = pd.Timestamp('20140701')
    days_delta = pd.Timedelta('30 days 00:00:00')
    df['days_since_last_used'] = today - df['date_last_trip']
    df['churn'] = df['days_since_last_used'] > days_delta

    #remove old date columns and last sugnup date to avoid leakage
    df = df.drop(['days_since_last_used', 'last_trip_date','signup_date'], axis=1)

    return df

def clean_data(df,constant_and_drop=False):

    #remove null vals for avg. rating by driver: only ~0% of total and non-defined phone
    remove_null_in_columns = ['avg_rating_by_driver','phone']

    for col in remove_null_in_columns:
        df = df[pd.notnull(df[col])]

    # Engineer a column for did they rate a driver
    df['rated_driver'] = df.loc[:,('avg_rating_of_driver')].apply(lambda x: pd.notnull(x))

    #handling with null values and dummifying columns
    df = impute_median(df, ['avg_rating_of_driver'])
    dummy_cols = ['phone', 'rated_driver', 'city']
    df = dummify(df, dummy_cols, constant_and_drop=constant_and_drop)

    return df

def impute_median(df, cols):
    '''
        Given a dataframe, find all the numeric types and impute the median values
        for any rows which are nan.

        return DataFrame -- the dataframe with all the nan values set to the imputed median
    '''
    for col in cols:
        median = df[col].median()
        df[col] = df[col].apply(lambda x: median if pd.isnull(x) else x)

    return df


def dummify(df, cols, constant_and_drop=False):
    '''
        Given a dataframe, for all the columns which are not numericly typed already,
        create dummies. This will NOT remove one of the dummies which is required for
        linear regression.

        returns DataFrame -- a dataframe with all non-numeric columns swapped into dummy columns
    '''
    df = pd.get_dummies(df, columns=cols, drop_first=constant_and_drop)
    if constant_and_drop:
        const = np.full(len(df), 1)
        df['constant'] = const

    return df

def clean_all(df, Time_series = False,constant_and_drop=False):
    '''
    Create a bool prediction column to indicated if churn == True (last used date > 30)
    '''
    copy = df.copy()
    copy = create_prediction(copy)
    copy = clean_data(copy,constant_and_drop)

    #create time series for churn statistics, optional
    churn_stats = copy.pop('date_last_trip')

    if Time_series:
        return copy, churn_stats
    return copy
