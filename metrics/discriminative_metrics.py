"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator


def discriminative_score_metrics (ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

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
    iterations = 2000 #test 500 # default = 2000
    batch_size = 128
    
    # Define the Discriminator model
    class Discriminator(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru = tf.keras.layers.GRU(hidden_dim, return_sequences=False)
            self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
    
        def call(self, x):
            h = self.gru(x)
            y_hat = self.dense(h)
            return y_hat
    
    # Instantiate the Discriminator
    discriminator = Discriminator(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    # Training loop using GradientTape
    for epoch in range(iterations):
        # Batch setting
        X_mb, _ = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, _ = batch_generator(train_x_hat, train_t_hat, batch_size)
    
        with tf.GradientTape() as tape:
            y_real = discriminator(X_mb)
            y_fake = discriminator(X_hat_mb)
    
            d_loss_real = loss_fn(tf.ones_like(y_real), y_real)
            d_loss_fake = loss_fn(tf.zeros_like(y_fake), y_fake)
            d_loss = d_loss_real + d_loss_fake
    
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, D_loss: {d_loss.numpy():.4f}")
    
    # Test performance on the testing set
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    test_x_hat = tf.convert_to_tensor(test_x_hat, dtype=tf.float32)
    
    y_pred_real = discriminator(test_x)
    y_pred_fake = discriminator(test_x_hat)
    
    y_pred_final = np.squeeze(np.concatenate((y_pred_real.numpy(), y_pred_fake.numpy()), axis=0))
    y_label_final = np.concatenate((np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))), axis=0)
    
    # Compute accuracy and discriminative score
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score
