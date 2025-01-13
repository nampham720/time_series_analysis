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

def process_time_series(df, grouping=True):
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
    freq_str = f'{most_frequent_diff}s'

    if grouping:
        # Initialize variables for finding ranges
        ranges = []
        start = df['timestamp'].iloc[0]
        
        # Iterate through the dataframe to find ranges
        for i in range(1, len(df)):
            if time_diff.iloc[i] > most_frequent_diff:
                end = df['timestamp'].iloc[i - 1]
                ranges.append((start, end))
                start = df['timestamp'].iloc[i]
        
        # Add the last range
        ranges.append((start, df['timestamp'].iloc[-1]))
        
        # Resample the dataset
        df_resampled = df.set_index('timestamp').resample(freq_str).agg({
            'value': ['min', 'max', 'mean']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        df_resampled.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_resampled.columns.values]
        
        # Filter resampled data to keep only those within the original ranges
        filtered_resampled = pd.DataFrame()
        for start, end in ranges:
            mask = (df_resampled['timestamp'] >= start) & (df_resampled['timestamp'] <= end)
            filtered_resampled = pd.concat([filtered_resampled, df_resampled[mask]])
        
        df = filtered_resampled.reset_index(drop=True)
    
    return most_frequent_diff, df


def period_pattern(df, time_diff):   
    '''
    Calculate the number of events 
    Input: 
    -df: dataframe
    -time_diff: the time inverval 
    Return:
    periods of dict type {window: time in seconds}
    ''' 
    period_per_day = int(24 * 3600 / time_diff)
    if time_diff <= 300:
        periods = {3: int(time_diff*3),
                   6: int(time_diff*6),
                   9: int(time_diff*9),
                   period_per_day: int(24*3600)}
    else:
        periods = {1: int(time_diff),
                   2: int(time_diff*2),
                   3: int(time_diff*3),
                   period_per_day: int(24*3600)}
    
    return periods


# Perform statistics to define trend, seasonality, residual
def analyze_time_series(df, periods, add_or_mul, value_of_choice):
    """
    Analyze the time series DataFrame.
    
    Parameters:
    - df: DataFrame containing the time series data.
    - periods: Dictionary with period names as keys and period values as values for decomposition.
    - value_of_choice: Column to analyze ('value_min', 'value_max', or 'value_mean').
    
    Returns:
    - results_df: DataFrame summarizing the characteristics of the time series for each period.
    """
    # Ensure 'timestamp' is a datetime object and set as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=[
        'period_name', 'trend_var', 'seasonal_var', 'residual_var',
        'residual_normality', 'trend', 'trend_slope', 'significant_seasonal'
    ])

    # Define time period columns for grouping
    period_columns = {'perday': 'D', 'perweek': 'W', 'permonth': 'M'}
    periods_list = list(periods.values())
    

    for i in range(len(periods)):
        try:
            # Decompose the time series
            decomposition = seasonal_decompose(df[value_of_choice], 
                                               model=add_or_mul[i], period=periods_list[i][0])
        except:
            continue
        
        # Extract components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Create a DataFrame with decomposition results
        decomposed_df = pd.DataFrame({
            'value': df[value_of_choice],
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        })

        # Variance calculations
        trend_variance = np.var(decomposed_df['trend'].dropna())
        seasonal_variance = np.var(decomposed_df['seasonal'].dropna())
        residual_variance = np.var(decomposed_df['residual'].dropna())
        
        # Residual normality test
        _, residual_p_value = shapiro(decomposed_df['residual'].dropna())
        residual_normality = residual_p_value > 0.05

        # Mann-Kendall trend test
        trend_result = mk.original_test(decomposed_df['value'])

        # Add grouping column for seasonality test
        #if periods_name[i] in period_columns:
        df[f'group_{periods_name[i]}'] = df.index.to_period(period_columns[periods_name[i]])
        
        # Kruskal-Wallis test for seasonality
        groups = [group[value_of_choice].values for _, group in df.groupby(f'group_{periods_name[i]}')]
        _, seasonal_p_value = kruskal(*groups)
        significant_seasonal = seasonal_p_value < 0.05
        
        # Create a new row for this period
        new_row = pd.DataFrame([{
            'period_name': periods_name[i],
            'trend_var': trend_variance,
            'seasonal_var': seasonal_variance,
            'residual_var': residual_variance,
            'residual_normality': residual_normality,
            'trend': trend_result.trend,
            'trend_slope': trend_result.slope,
            'significant_seasonal': significant_seasonal
        }])

        # Check if new_row contains only NA values or is empty
        if not new_row.isna().all(axis=None):
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
    # Reset index to maintain consistency
    df.reset_index(inplace=True)
    
    return results_df

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



# Define multiplicative or additive
# Define multiplicative or additive
def calculate_dynamic_threshold(variance_reductions, k=1.0):
    mean_var_reduction = np.mean(variance_reductions)
    std_var_reduction = np.std(variance_reductions)
    return mean_var_reduction + k * std_var_reduction

def calculate_variance_reduction(df, p, value_of_choice):
    segment_size = p
    variance_reductions = []

    for start in range(0, len(df) - segment_size + 1, segment_size):
        segment = df[value_of_choice].iloc[start:start + segment_size]
        log_segment = np.log(segment.replace(0, np.nan).dropna())

        original_var = segment.var()
        log_var = log_segment.var()

        if original_var > 0:  # Avoid division by zero
            variance_reduction = (original_var - log_var) / original_var
            variance_reductions.append(variance_reduction)

    return variance_reductions

def add_or_mul(df, periods, value_of_choice):
    period_values = list(periods.keys())
    final_results = []
    
    for p in period_values:
        try:
            # Step 1: Rolling Mean-Variance Correlation
            rolling_mean = df[value_of_choice].rolling(window=p).mean()
            rolling_var = df[value_of_choice].rolling(window=p).var()
            correlation = rolling_mean.corr(rolling_var)
            
            rolling_result = 1 if correlation > 0.5 else 0  # 1 = Multiplicative, 0 = Additive
            
            # Step 2: Log Transformation Variance Test
            variance_reductions = calculate_variance_reduction(df, p, value_of_choice)
    
            # Calculate dynamic threshold
            dynamic_threshold = calculate_dynamic_threshold(variance_reductions)
            
            # Step 3: Exponential Smoothing Model 
            # Exponential for multiplicative requires positives value, shifting must be done nonetheless. 
            # Shift data to be positive
            shift_constant = abs(df[value_of_choice].min()) + 1
            df['shifted_value'] = df[value_of_choice] + shift_constant
            add_model = ExponentialSmoothing(df['shifted_value'], trend='add', seasonal='add', seasonal_periods=30).fit()
            mul_model = ExponentialSmoothing(df['shifted_value'], trend='mul', seasonal='mul', seasonal_periods=30).fit()
            
            # Calculate RMSE for both models
            add_rmse = np.sqrt(mean_squared_error(df['shifted_value'], add_model.fittedvalues))
            mul_rmse = np.sqrt(mean_squared_error(df['shifted_value'], mul_model.fittedvalues))
            
            smoothing_result = 1 if mul_rmse < add_rmse else 0  # 1 = Multiplicative, 0 = Additive
    
            # Step 4: Combine Results
            log_result = 1 if np.mean(variance_reductions) > dynamic_threshold else 0
            final_score = rolling_result + log_result + smoothing_result
            final_result = "Multiplicative" if final_score >= 2 else "Additive"
            final_results.append(final_result)
        except Exception as e:
            final_results.append('none')
    return final_results