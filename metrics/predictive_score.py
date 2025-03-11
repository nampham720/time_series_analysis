import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential

def predictive_score_metrics(ori_data, generated_data, epochs=50, batch_size=128, n_splits=5):
    """Evaluate Post-hoc RNN one-step-ahead prediction with improvements.
    
    Args:
        - ori_data: Original time-series data
        - generated_data: Generated synthetic data
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        - n_splits: Number of splits for cross-validation
        
    Returns:
        - predictive_score: Mean MAE of predictions on the original data
    """

    # Convert to NumPy arrays
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)

    # Extract shape parameters
    no, seq_len, dim = ori_data.shape

    # Prepare the data for training and testing
    def prepare_data(data):
        """Prepares input-output pairs for the predictor."""
        X = np.array([d[:-1, -1:] for d in data])  # Keep only the last column (univariate time series)
        Y = np.array([d[1:, -1:] for d in data])  # The next time step prediction
        return X, Y

    # Prepare data for training
    X_ori, Y_ori = prepare_data(ori_data)
    X_gen, Y_gen = prepare_data(generated_data)

    # Define the model architecture
    def build_model(hidden_dim):
        """Build a simple RNN model."""
        model = Sequential()
        model.add(Bidirectional(GRU(hidden_dim, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
        model.add(Dropout(0.2))  # Dropout layer to reduce overfitting
        model.add(Dense(1, activation=None))  # Output layer with no activation
        model.compile(optimizer='adam', loss='mae')  # Mean Absolute Error Loss
        return model

    # Cross-validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []

    # Cross-validation loop
    for train_index, val_index in kf.split(X_ori):
        X_train, X_val = X_ori[train_index], X_ori[val_index]
        Y_train, Y_val = Y_ori[train_index], Y_ori[val_index]
        
        # Create the model
        predictor = build_model(hidden_dim=max(1, int(dim / 2)))  # Hidden dimension is half the input dimension
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        predictor.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                      validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=1)
        
        # Predict on the validation set (original data)
        pred_Y = predictor.predict(X_val)

        Y_val = Y_val.reshape(-1)  # Flatten to a 1D array
        pred_Y = pred_Y.reshape(-1)  # Flatten to a 1D array

        # Calculate MAE for this fold
        mae_score = mean_absolute_error(Y_val, pred_Y)
        mae_scores.append(mae_score)

    # Compute mean MAE across all folds
    predictive_score = np.mean(mae_scores)

    return predictive_score
