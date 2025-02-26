import pandas as pd 
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller (ADF) Test
def adf_test(series):
    result = adfuller(series)
    if result[1] <= 0.05:
        return True #print("The series is stationary (reject H0)")
    else:
        return False #print("The series is non-stationary (fail to reject H0)")


# Define multiplicative or additive
def add_or_mul(df, window_sizes):
    window_freq = list(window_sizes.values())
    value_of_choices = ['smoothed_first', 'smoothed_daily']
    final_results = []

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

    for p, value_of_choice in zip(window_freq, value_of_choices):
        value_of_choice_result = list()
        try:
            # Step 1: Rolling Mean-Variance Correlation
            rolling_mean = df[value_of_choice]
            rolling_var = df[value_of_choice]
            correlation = rolling_mean.corr(rolling_var)
            
            rolling_result = 1 if correlation > 0.5 else 0  # 1 = Multiplicative, 0 = Additive
            
            # Step 2: Log Transformation Variance Test
            variance_reductions = calculate_variance_reduction(df, p, value_of_choice)
    
            # Calculate dynamic threshold
            dynamic_threshold = calculate_dynamic_threshold(variance_reductions)
            log_result = 1 if np.mean(variance_reductions) > dynamic_threshold else 0

            # Step 3: Combine Results
            final_score = rolling_result + log_result
            final_result = "Multiplicative" if final_score >= 2 else "Additive"
            final_results.append(final_result)
        except Exception as e:
            final_results.append('none')
        
    return final_results


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
    periods_time= list(periods.values())
    periods_window = list(periods.keys())

    for i in range(len(periods)):
        groupname = f'per_{periods_time[i] / 3600}hr'
        try:
            # Decompose the time series
            decomposition = seasonal_decompose(df[value_of_choice], 
                                               model=add_or_mul[i], period=periods_window[i])
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

        # Kruskal-Wallis test for seasonality
        groups = [group.dropna() for _, group in seasonal.groupby(seasonal.index)]
        # Perform Kruskal-Wallis test
        stat, p_value = kruskal(*groups)
        significant_seasonal = seasonal_p_value < 0.05
        
        # Create a new row for this period
        new_row = pd.DataFrame([{
            'period_name': groupname,
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