import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import extract_time 
from tensorflow.keras.callbacks import LambdaCallback

def train_test_divide(ori_data, generated_data, ori_time, generated_time):
    # Split original and generated data into training and testing sets
    train_x, test_x, train_t, test_t = train_test_split(ori_data, ori_time, test_size=0.2, random_state=42)
    train_x_hat, test_x_hat, train_t_hat, test_t_hat = train_test_split(generated_data, generated_time, test_size=0.2, random_state=42)
    
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

def discriminative_score_metrics(ori_data, generated_data, iterations=200, batch_size=128, patience=10):
    """Use post-hoc RNN to classify original data and synthetic data with optimizations."""
    
    ori_data = np.array(ori_data)
    generated_data = np.array(generated_data)

    # Get shape parameters
    no, seq_len, dim = ori_data.shape
    
    # Extract sequence lengths
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)
    
    # Network parameters
    hidden_dim = max(1, int(dim / 2))

    class Discriminator(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru1 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
            self.gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=False)
            self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, x):
            # Ensure input is 3D (batch_size, seq_len, feature_dim)
            if len(x.shape) == 2:  # If the input is 2D, reshape it to 3D
                x = tf.expand_dims(x, axis=0)  # Add batch dimension

            h = self.gru1(x)
            h = self.gru2(h)
            y_hat = self.dense(h)
            return y_hat

    # Instantiate the Discriminator
    discriminator = Discriminator(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Reshape the data to have 3 dimensions (batch_size, seq_len, features)
    train_x = train_x.reshape(-1, seq_len, 1)
    train_x_hat = train_x_hat.reshape(-1, seq_len, 1)
    test_x = test_x.reshape(-1, seq_len, 1)
    test_x_hat = test_x_hat.reshape(-1, seq_len, 1)

    # Create train_labels for the real and fake data
    train_labels_real = np.ones((train_x.shape[0], 1))
    train_labels_fake = np.zeros((train_x_hat.shape[0], 1))
    train_x_combined = np.concatenate((train_x, train_x_hat), axis=0)
    train_labels_combined = np.concatenate((train_labels_real, train_labels_fake), axis=0)

    # Shuffle data
    indices = np.arange(train_x_combined.shape[0])
    np.random.shuffle(indices)
    train_x_combined = train_x_combined[indices]
    train_labels_combined = train_labels_combined[indices]

    # Combine test data
    test_labels_real = np.ones((test_x.shape[0], 1))
    test_labels_fake = np.zeros((test_x_hat.shape[0], 1))
    test_x_combined = np.concatenate((test_x, test_x_hat), axis=0)
    test_labels_combined = np.concatenate((test_labels_real, test_labels_fake), axis=0)


    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x_combined, train_labels_combined)).batch(batch_size).repeat()

    # Early stopping setup
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    
    # Compile the model
    discriminator.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    discriminator.fit(
        train_dataset,  # The training data
        epochs=iterations,  # Number of training epochs
        steps_per_epoch=len(train_x_combined) // batch_size,  # Define number of steps per epoch
        validation_data=(test_x_combined, test_labels_combined),  # Validation data
        callbacks=[early_stopping]  # Early stopping and custom epoch display
    )

    # Predict on test set
    y_pred_real = discriminator(test_x)
    y_pred_fake = discriminator(test_x_hat)

    y_pred_final = np.squeeze(np.concatenate((y_pred_real.numpy(), y_pred_fake.numpy()), axis=0))
    y_label_final = np.concatenate((np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))), axis=0)

    # Compute accuracy and discriminative score
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
