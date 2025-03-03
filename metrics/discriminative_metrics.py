import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator

def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data

    Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
    """

    # Get shape parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    
    # Extract sequence lengths
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)
    
    # Network parameters
    hidden_dim = max(1, int(dim / 2))
    iterations = 200  # Default 2000, testing 200
    batch_size = 128
    
    # Define the Discriminator model
    class Discriminator(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru1 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
            self.gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=False)
            self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, x):
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

    # Convert test data to tensors once (to avoid repeated conversions)
    test_x_tensor = tf.convert_to_tensor(test_x, dtype=tf.float32)
    test_x_hat_tensor = tf.convert_to_tensor(test_x_hat, dtype=tf.float32)

    # Training loop
    for epoch in range(iterations):
        # Generate batch once per iteration
        X_mb_real, _ = batch_generator(train_x, train_t, batch_size // 2)
        X_mb_fake, _ = batch_generator(train_x_hat, train_t_hat, batch_size // 2)
        
        # Concatenate real and fake data in a single batch
        X_mb = np.vstack((X_mb_real, X_mb_fake))
        y_mb = np.vstack((np.ones((batch_size // 2, 1)), np.zeros((batch_size // 2, 1))))

        with tf.GradientTape() as tape:
            y_pred = discriminator(X_mb)
            d_loss = loss_fn(y_mb, y_pred)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, D_loss: {d_loss.numpy():.4f}")

    # Predict on test set
    y_pred_real = discriminator(test_x_tensor)
    y_pred_fake = discriminator(test_x_hat_tensor)

    y_pred_final = np.squeeze(np.concatenate((y_pred_real.numpy(), y_pred_fake.numpy()), axis=0))
    y_label_final = np.concatenate((np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))), axis=0)

    # Compute accuracy and discriminative score
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
