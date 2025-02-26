import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator

def timegan(ori_data, parameters):
    """TimeGAN function (TensorFlow 2.18 Compatible)."""
    tf.keras.backend.clear_session()
    no, f, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data -= min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    
    def build_rnn(module_name, hidden_dim, num_layers):
        layers = [
            tf.keras.layers.RNN(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)],
                return_sequences=True
            ),
            tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
        ]
        return tf.keras.Sequential(layers)

    embedder = build_rnn(module_name, hidden_dim, num_layers)
    recovery = build_rnn(module_name, dim, num_layers)
    generator = build_rnn(module_name, hidden_dim, num_layers)
    supervisor = build_rnn(module_name, hidden_dim, num_layers - 1)
    discriminator = tf.keras.Sequential([
        tf.keras.layers.RNN(
            [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)],
            return_sequences=True
        ),
        tf.keras.layers.Dense(1, activation=None)
    ])
    
    optimizer = tf.keras.optimizers.Adam()
    
    #@tf.function
    def train_step(X_mb, T_mb):
        with tf.GradientTape(persistent=True) as tape:
            H = embedder(X_mb)
            X_tilde = recovery(H)
            
            # Convert T_mb to a tensor here
            T_mb_tensor = tf.convert_to_tensor(T_mb)
            
            # Call random_generator with T_mb_tensor
            Z_mb = random_generator(batch_size, z_dim, T_mb_tensor, max_seq_len)
            
            E_hat = generator(Z_mb)
            H_hat = supervisor(E_hat)
            X_hat = recovery(H_hat)
            X_mb = tf.cast(X_mb, dtype=tf.float32)
            Y_fake = tf.cast(discriminator(H_hat), dtype=tf.float32)
            Y_real = tf.cast(discriminator(H), dtype=tf.float32)
            
            D_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(Y_real, dtype=tf.float32), Y_real
            ) + tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.zeros_like(Y_fake, dtype=tf.float32), Y_fake
            )
            
            G_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(Y_fake, dtype=tf.float32), Y_fake
            ) + tf.reduce_mean(tf.abs(X_hat - X_mb))

        d_grads = tape.gradient(D_loss, discriminator.trainable_variables)
        g_grads = tape.gradient(G_loss, generator.trainable_variables + supervisor.trainable_variables)
        # Check if d_grads and g_grads are tensors, not numpy arrays
        #print("Checking d_grads and g_grads types:")
        #for grad in d_grads:
        #    print(type(grad))  # Should be <class 'tensorflow.python.framework.ops.EagerTensor'>
        #for grad in g_grads:
        #    print(type(grad))  # Should be <class 'tensorflow.python.framework.ops.EagerTensor'>
        
        optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        #optimizer.apply_gradients(zip(g_grads, generator.trainable_variables + supervisor.trainable_variables))
        # Create a new optimizer for the generator and supervisor variables
        generator_and_supervisor_optimizer = tf.keras.optimizers.Adam()
    
        # Apply the gradients for the generator and supervisor variables
        generator_and_supervisor_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables + supervisor.trainable_variables)
        )

        #return d_grads, g_grads, discriminator.trainable_variables, generator.trainable_variables, supervisor.trainable_variables
    for epoch in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        #d_grads, g_grads, discriminator, generator, supervisor = train_step(X_mb, T_mb)
        #if epoch % 1000 == 0:
        #    print(f"Epoch {epoch}: Training step completed")
    #return d_grads, g_grads, discriminator, generator, supervisor
    print('Training completed')
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data_curr = recovery(supervisor(generator(Z_mb)))
    generated_data = [(generated_data_curr[i, :ori_time[i], :] * max_val) + min_val for i in range(no)]
    return generated_data
    