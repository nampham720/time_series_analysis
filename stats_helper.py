import pandas as pd 
from statsmodels.tsa.stattools import adfuller
import numpy as np 

# Augmented Dickey-Fuller (ADF) Test
def adf_test(series):
    if len(series) < 10:  # Ensure at least 10 data points
        return False, False
    result = adfuller(series)
    if result[1] <= 0.05:
        return True, False #print("The series is stationary (reject H0)"), if stationary then trend does not exist
    else:
        return False, True #print("The series is non-stationary (fail to reject H0)")

def seasonality_test(series, model, period):
    # 2. **Check for Seasonality using Seasonal Decomposition**
    if period < 2 or len(series) < period:
        return False  # Not enough data for decomposition
    try:
        decomposition = seasonal_decompose(series, model=model, period=period)
        seasonal_strength = np.std(decomposition.seasonal) / np.std(series)
        seasonality_present = seasonal_strength > 0.1  # If seasonal component is significant
        return seasonality_present
    except:
        return False  # If decomposition fails, assume no seasonality


# Define multiplicative or additive
def add_or_mul(df, window_sizes):
    window_freq = list(window_sizes.values()) #return daily freq and time_of_day freq
    value_of_choice = 'value'
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

    for p in window_freq:
        
        #try:
        # Step 1: Rolling Mean-Variance Correlation
        rolling_mean = df[value_of_choice].rolling(window=p).mean()
        rolling_var = df[value_of_choice].rolling(window=p).var()

        # Ensure NaN values are dropped
        valid_data = pd.concat([rolling_mean, rolling_var], axis=1).dropna()
        
        # Compute correlation safely
        correlation = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
        
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
        #except Exception as e:
        #    final_results.append('none')
        
    return final_results


def initial_stats(params):
    df          = params['df'].copy()
    periods     = params['periods'] 
    model       = params['model']
    
    model_1, model_2   = model.iloc[0], model.iloc[1]
    period_1, period_2 = periods['time_of_day'], periods['daily']

    stationary, trend = adf_test(df['value'])
    p1_seasonality  = seasonality_test(df['value'], model_1, period_1) 
    p2_seasonality  = seasonality_test(df['value'], model_2, period_2)

    
    return [stationary, trend, p1_seasonality, p2_seasonality]

