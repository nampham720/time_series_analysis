"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import numpy as np
import tensorflow as tf


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]      

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
    """Returns Maximum sequence length and each sequence length.

    Args:
    - data: original data

    Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))

    return time, max_seq_len

#updated rnn_cell
def rnn_cell(module_name, hidden_dim):
    """Creates an RNN cell based on the given module type.

    Args:
        - module_name (str): One of 'gru', 'lstm', or 'lstmLN' (LayerNorm LSTM).
        - hidden_dim (int): Number of hidden units.

    Returns:
        - rnn_cell (tf.keras.layers.Layer): The corresponding RNN cell.
    """
    assert module_name in ['gru', 'lstm', 'lstmLN'], "Invalid module_name. Choose from ['gru', 'lstm', 'lstmLN']."

    if module_name == 'gru':
        return tf.keras.layers.GRUCell(hidden_dim, activation='tanh')
    elif module_name == 'lstm':
        return tf.keras.layers.LSTMCell(hidden_dim, activation='tanh')
    elif module_name == 'lstmLN':
        return tf.keras.layers.LSTMCell(hidden_dim, activation='tanh', recurrent_activation='sigmoid', use_layer_norm=True)

    return None
#def rnn_cell(module_name, hidden_dim):
#  """Basic RNN Cell.
#    
#  Args:
#    - module_name: gru, lstm, or lstmLN
#    
#  Returns:
#    - rnn_cell: RNN Cell
#  """
#  assert module_name in ['gru','lstm','lstmLN']
#  
#  # GRU
#  if (module_name == 'gru'):
#    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
#  # LSTM
#  elif (module_name == 'lstm'):
#    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
#  # LSTM Layer Normalization
#  elif (module_name == 'lstmLN'):
#    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
#  return rnn_cell


def random_generator (batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length

    Returns:
    - Z_mb: generated random vector
    """
    # Initialize Z_mb as a tensor of zeros with dtype tf.float32
    Z_mb = tf.zeros([batch_size, max_seq_len, z_dim], dtype=tf.float32)

    # Ensure T_mb is of type int32 (cast if necessary)
    T_mb = tf.cast(T_mb, dtype=tf.int32)

    # Generate random values for each time step per sequence
    for i in range(batch_size):
        time_steps = T_mb[i]  # Get the time steps for the i-th sequence
        
        # Generate random values for the given time steps
        temp_Z = tf.random.uniform([time_steps, z_dim], minval=0.0, maxval=1.0, dtype=tf.float32)

        # Create an index for each element of the sequence to update Z_mb
        indices = tf.stack([tf.fill([time_steps], i), tf.range(time_steps)], axis=1)

        # Scatter the random vector into the corresponding positions in Z_mb
        Z_mb = tf.tensor_scatter_nd_update(Z_mb, indices, temp_Z)

    #Z_mb = list()
    #for i in range(batch_size):
    #  temp = np.zeros([max_seq_len, z_dim])
    #  temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    #  temp[:T_mb[i],:] = temp_Z
    #  Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch

    Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]     
            
    #X_mb = list(data[i] for i in train_idx)
    #T_mb = list(time[i] for i in train_idx)
    # Convert the list to np.array
    X_mb = np.array([data[i] for i in train_idx], dtype=np.float32)  # Convert to NumPy array
    T_mb = np.array([time[i] for i in train_idx])  # Convert to NumPy array
    
    X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
    return X_mb, T_mb

