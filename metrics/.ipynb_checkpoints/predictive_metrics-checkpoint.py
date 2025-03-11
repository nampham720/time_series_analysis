"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time

 
def predictive_score_metrics(ori_data, generated_data):
    """Evaluate Post-hoc RNN one-step-ahead prediction.
    
    Args:
        - ori_data: Original time-series data
        - generated_data: Generated synthetic data
        
    Returns:
        - predictive_score: MAE of predictions on the original data
    """

    # Convert to NumPy arrays
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)

    # Extract shape parameters
    no, seq_len, dim = ori_data.shape

    # Extract time sequences
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    # Model parameters
    hidden_dim = max(1, int(dim / 2))
    epochs = 50 #50  # Reduced iterations for TF2 efficiency
    batch_size = 128

    # Data preparation function
    def prepare_data(data, time):
        """Prepares input-output pairs for the predictor."""
        #X = np.array([d[:-1, :-1] for d in data])
        X = np.array([d[:-1, 1:] for d in data])  # Keep value column, remove timestamps only
        T = np.array([t - 1 for t in time])
        #Y = np.array([np.reshape(d[1:, -1], (-1, 1)) for d in data])
        Y = np.array([d[1:, 1:] for d in data])  # Ensure Y is aligned with X
        

        return X, T, Y

    # Prepare training data
    train_X, train_T, train_Y = prepare_data(generated_data, generated_time)
    print('DONE PREPARE_DATA')
    # Define the Predictor model using Keras API
    class Predictor(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Predictor, self).__init__()
            self.rnn = tf.keras.layers.GRU(hidden_dim, activation='tanh', return_sequences=True)
            self.dense = tf.keras.layers.Dense(1, activation=None)

        def call(self, x):
            rnn_out = self.rnn(x)
            return tf.nn.sigmoid(self.dense(rnn_out))

    # Initialize the predictor model
    predictor = Predictor(hidden_dim)

    # Loss and Optimizer
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam()

    # Convert dataset into TensorFlow dataset API for efficiency
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    #train_dataset = train_dataset.shuffle(len(train_X)).batch(batch_size)
    train_dataset = train_dataset.shuffle(len(train_X)).batch(batch_size, drop_remainder=True)

    # Training loop using GradientTape
    for epoch in range(epochs):
        if epoch % 50 == 0:
            print('Epoch: ', epoch)
        for X_batch, Y_batch in train_dataset:
            print(X_batch.shape())
            with tf.GradientTape() as tape:
                Y_pred = predictor(X_batch)
                loss = loss_fn(Y_batch, Y_pred)

            grads = tape.gradient(loss, predictor.trainable_variables)
            optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

    # Prepare test data (on original dataset)
    test_X, test_T, test_Y = prepare_data(ori_data, ori_time)

    # Prediction on original data
    pred_Y = predictor(test_X).numpy()

    # Compute MAE
    mae_scores = [mean_absolute_error(test_Y[i], pred_Y[i]) for i in range(no)]
    predictive_score = np.mean(mae_scores)

    return predictive_score