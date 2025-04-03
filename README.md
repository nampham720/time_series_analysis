# time_series_analysis
# Thesis Work  
**Title:** Exploring ARIMA and TimeGAN Combined with ECOD for Anomaly Detection in Time Series: A Focus on RealKnownCause in the NAB Dataset  
## Process
Exploring how to use ARIMA and TimeGAN to capture temporal dependencies. 

Once done, calculate the residuals between the predicted value and original one. 

Apply ECOD based on the residuals to see which one works better in terms of capturing true outliers. 

The process is visualized as below: 

![Process](process.png)

The sequence length of this is on a daily basis. 

## Repo structure:
- Image/: final results
- timeganlogs/: The best combinationa of hyperparameters obtained through Optuna for TimeGAN
- realKnơnCause/: NAB datasets
- metrics/: refactored metrics to work with tensorflow v2.18. 
- anomaly_categorize.py: Define types of anomaly - point, contextual, collective - based on preliminary assumption. 
- stats_helper.py: Script contains functions to find stationarity, trend, seasonality components. 
- preprocess_functions.py: Script contains preprocess functions. 


## References 
NAB dataset: [Source](https://github.com/numenta/NAB)

The code of TimeGAN is originally provided by Yoon et al. - [Source](https://github.com/jsyoon0823/TimeGAN)
* Note that the code has been refactored to work with latest tensorflow v2.18. The original code was written in the old version. 
 