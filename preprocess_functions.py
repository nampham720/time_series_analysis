import pandas as pd 
from pathlib import Path
import json 
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kruskal
import pymannkendall as mk 
from glob import glob
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

def get_all_files(path):
    '''Get all the files from given path
    path: path to file
    '''
    allfiles = [str(file) for file in Path(path).rglob('*') if file.is_file()]
    return [file for file in allfiles if '.ipynb' not in file ]


def read_csv_file(path):
    '''
    Read a file based on given path
    path: path to file 
    '''
    
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df 

def retrieve_error_window(error_path, file_names):
    '''Read a json file based on given path
    path: path to file

    Return: a list of error_windows
    '''
    with open(error_path, "r") as file:
        errors = json.load(file)

    converted_errors = list()
    for filename in file_names:
        df_error = errors.get(filename)

        # Convert error windows to datetime
        error_windows = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in df_error]
        converted_errors.append(error_windows)

    return converted_errors


def is_outlier(timestamp, error_windows):
    ''' Mark outliers based on error_windows
    timestamp: timestamp from dataset
    error_windows: known window outliers
    Return: 1 if outlier, else 0
    ''' 
    for start, end in error_windows:
        if start <= timestamp <= end:
            return 1
    return 0

def process_time_series(PATH, error_file):
    df = pd.read_csv(PATH, parse_dates=['timestamp'])
    error_list = error_file.get(PATH)
    df_errors = pd.DataFrame({'timestamp': pd.to_datetime(error_list)})
    df_errors['outlier'] = 1
    
    # Ensure the timestamps are sorted
    if not df['timestamp'].is_monotonic_increasing:
        df = df.sort_values(by='timestamp')
    
    # Calculate time differences
    time_diff = df['timestamp'].diff().dt.total_seconds()
    
    # Count the occurrences of each time difference
    time_diff_counts = time_diff.value_counts()
    
    # Get the most frequent time difference
    most_frequent_diff = time_diff_counts.idxmax()
    
    # Convert time_diff to a frequency string
    freq_str = f'{int(most_frequent_diff)}s'

    
    # Get first and last timestamps
    first_timestamp = df['timestamp'].min()
    last_timestamp = df['timestamp'].max()
    
    # Create a complete range of timestamps from first to last
    full_range = pd.date_range(start=first_timestamp, end=last_timestamp, freq=freq_str)
    
    # Convert to set to identify differences
    full_range_set = set(full_range)
    original_set = set(df['timestamp'])
    
    # Find missing timestamps
    differences = original_set - full_range_set
    
    # If missing timestamps, add them to the full range
    if len(differences) > 0:
        full_range_set.update(differences)
    
    # Convert the full range back to a sorted DataFrame
    df_full = pd.DataFrame({'timestamp': sorted(list(full_range_set))})
    
    # Merge original data, keeping existing values
    df_resampled = pd.merge(df_full, df, on='timestamp', how='left')
    
    # Interpolate missing values
    df_resampled['value'] = df_resampled['value'].interpolate(method='linear')
    
    # Add a new column that indicates the interval start time (using freq_str)
    df_resampled['interval_start'] = df_resampled['timestamp'].apply(
        lambda x: first_timestamp + pd.Timedelta(seconds=(x - first_timestamp).total_seconds() // most_frequent_diff * most_frequent_diff)
    )
    
    # Merge with error outlier data
    df_resampled = df_resampled.merge(df_errors, on='timestamp', how='left')

    # Reorganize columns and group by the 5-minute intervals
    df_resampled = df_resampled[['interval_start', 'value', 'outlier']] 
    df_resampled = df_resampled.groupby('interval_start').mean().reset_index()

    # Fill missing outlier values with 0 and apply condition for outliers
    df_resampled['outlier'] = df_resampled['outlier'].fillna(0)
    df_resampled['outlier'] = df_resampled['outlier'].apply(lambda x: 1 if x > 0 else 0)

    # Rename the interval_start column to timestamp
    df = df_resampled.rename(columns={"interval_start": "timestamp"})

    return most_frequent_diff, df


def window_sizes_freq(time_diff):
    window_sizes = dict()
    window_sizes['daily'] = int(60*60*24 / time_diff)
    window_sizes['time_of_day'] = int(window_sizes['daily'] / 4)
    return window_sizes 


def define_skewness(df, value_of_choice):
    '''
    Define the skewness of value_of_choice
    Input:
    - df: dataframe
    - value_of_choice: value_min, max, or mean
    Return:
    -skewness: positive, negative, or symmetric 
    '''     
    # Calculate skewness using pandas
    skewness = df[value_of_choice].skew()
    
    # Interpret the result
    if skewness > 0:
        skew_type = 'positive'
    elif skewness < 0:
        skew_type = 'negative'
    else:
        skew_type = 'symmetric'
    
    return skew_type

def ratio_outlier(df):
    ''' 
    Calculate the ratio of outliers against non-outliers
    Input:
    -df : dataframe
    Return:
    The ratio 
    '''
    # Count the occurrences of each outlier value
    counts = df['outlier'].value_counts()
    
    # Calculate the ratio
    ratio = counts.get(1, 0) / counts.get(0, 1)  # Use default 0 for missing values
    return ratio


def ratio_na(df, value_of_choice):
    count_na = df[value_of_choice].isnull().sum()
    count_notna = df[value_of_choice].notnull().sum()
    return round(count_na / count_notna, 2)



