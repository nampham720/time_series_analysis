import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances


def categorize_anomalies(data, window_size=288, filename='unknown', z_thresh=2, max_comparisons=4):
    """
    Categorizes labeled anomalies in NAB datasets into Point, Contextual, or Collective using batch-based similarity measures
    for collective anomaly detection.

    Parameters:
        data (pd.DataFrame): Must contain 'timestamp', 'value', and 'outlier'.
        window_size (int): Batch size for collective anomaly detection.
        z_thresh (float): Z-score threshold for point anomaly detection.
        similarity_thresh (float): Threshold for similarity to classify a batch as anomalous.
        max_comparisons (int): Number of previous batches to compare with the current batch.

    Returns:
        pd.DataFrame: DataFrame with 'anomaly_type' column.
    """

    # Ensure sorted time series
    data = data.sort_values(by='timestamp').reset_index(drop=True)

    # **Point Anomalies using Z-score**
    data['z_score'] = np.abs(zscore(data['value']))
    data['is_point'] = (data['z_score'] > z_thresh).fillna(False)

    # **Contextual Anomalies using Rolling Statistics**
    rolling_mean = data['value'].rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = data['value'].rolling(window=window_size, center=True, min_periods=1).std()
    data['is_contextual'] = ((np.abs((data['value'] - rolling_mean) / rolling_std) > 1)
                              .fillna(False))  # Fill NaNs with False

    # **Collective Anomalies using Batch-Based Euclidean Distance**
    data['anomaly_type'] = 'Normal'  # Default to Normal for all rows
    scaler = MinMaxScaler()  # Scale values between 0 and 1
    data['value_normalized'] = scaler.fit_transform(data[['value']])

    batch_distances = []  # Store batch distances for plotting
    batch_timestamps = []  # Store timestamps for batch comparisons
    outlier_batches = []  # Track batches containing actual outliers

    num_batches = len(data) // window_size  # Number of full batches

    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * window_size
        batch_data = data.iloc[start_idx:start_idx + window_size]

        # Extract batch features (normalized values)
        batch_features = batch_data[['value_normalized']].values

        # Compare with previous batches
        similarities = []
        for prev_batch_idx in range(max(0, batch_idx - max_comparisons), batch_idx):
            prev_start_idx = prev_batch_idx * window_size
            prev_batch_data = data.iloc[prev_start_idx:prev_start_idx + window_size]
            prev_batch_features = prev_batch_data[['value_normalized']].values

            # Compute Euclidean distance between batches
            similarity = np.linalg.norm(batch_features - prev_batch_features)
            similarities.append(similarity)

        # Compute the average distance for this batch
        batch_distance = np.mean(similarities) if similarities else 0
        batch_distances.append(batch_distance)
        batch_timestamps.append(batch_data['timestamp'].min())  # Use the min timestamp of the batch

        # Check if the distance exceeds the threshold
        # Set threshold as the 95th percentile of last 4 distances (if available)
        if len(similarities) >= max_comparisons:
            similarity_thresh = np.percentile(similarities, 95)  
        else:
            similarity_thresh = np.mean(similarities) if similarities else 0  
            
        if len(similarities) > 0 and batch_distance > similarity_thresh:
            data.loc[start_idx:start_idx + window_size - 1, 'anomaly_type'] = 'Collective'
        
        # Mark batches containing actual outliers
        if any(batch_data['outlier'] == 1):
            outlier_batches.append(batch_idx)


    # **Assign Anomaly Type for Point and Contextual**
    def classify(row):
        if row['outlier'] == 1:
            if row['is_point']:
                return 'Point'
            elif row['is_contextual']:
                return 'Contextual'
            else:
                return 'Unknown'
        return row['anomaly_type']

    data['anomaly_type'] = data.apply(classify, axis=1)

    # **Visualizations**
    plt.figure(figsize=(20, 6))

    # **Plot 1: Z-Score Distribution**
    plt.subplot(1, 3, 1)
    sns.histplot(data['z_score'], bins=50, kde=True, color='skyblue')
    plt.axvline(z_thresh, color='red', linestyle='dashed', linewidth=2, label=f'Z-score threshold = {z_thresh}')
    plt.axvline(-z_thresh, color='red', linestyle='dashed', linewidth=2)
    
    # Highlight **actual outliers** in red
    outlier_z_scores = data[data['outlier'] == 1]['z_score']
    plt.scatter(outlier_z_scores, np.zeros(len(outlier_z_scores)), color='red', label='Actual Outliers', alpha=1)

    plt.title("Z-Score Distribution")
    plt.xlabel("Z-Score")
    plt.ylabel("Frequency")
    plt.legend()

    # **Plot 2: Contextual Anomalies (Rolling Window)**
    plt.subplot(1, 3, 2)
    plt.plot(data['timestamp'], data['value'], label='Value', color='skyblue', alpha=0.6, zorder=1)
    plt.plot(data['timestamp'], rolling_mean, label='Rolling Mean', linestyle='dashed', color='gray', zorder=2)
    plt.fill_between(data['timestamp'], rolling_mean - 1 * rolling_std, 
                     rolling_mean + 1 * rolling_std, color='gray', alpha=0.3, label="± Std Dev")

    # Highlight **actual outliers** in red
    plt.scatter(data[data['outlier'] == 1]['timestamp'], 
            data[data['outlier'] == 1]['value'], 
            color='red', label='Actual Outliers', marker='x', s=80, 
            linewidth=1.5, zorder=3)

    plt.title("Contextual Anomalies (Rolling Mean & Std Dev)")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()

    # **Plot 3: Batch-Based Euclidean Distance**
    plt.subplot(1, 3, 3)
    plt.plot(batch_timestamps, batch_distances, label="Batch Euclidean Distance", color='skyblue', zorder=1)
    
    # Highlight batches containing actual outliers
    outlier_distances = [batch_distances[i] for i in outlier_batches]
    outlier_times = [batch_timestamps[i] for i in outlier_batches]
    plt.scatter(outlier_times, outlier_distances, color='red', label="Outlier Batches", marker='x', s=100, zorder=2)
    
    plt.title("Collective Anomalies (Batch Euclidean Distance)")
    plt.xlabel("Timestamp (Min of Batch)")
    plt.ylabel("Euclidean Distance")

    plt.suptitle(filename)
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.savefig(f'images/outliers/{filename}.png')
    plt.show()

    
    # Drop temp columns before returning
    return data.drop(columns=['z_score', 'is_point', 'is_contextual', 'value_normalized'])