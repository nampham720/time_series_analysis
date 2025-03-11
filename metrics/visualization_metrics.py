"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

def visualization(ori_data, generated_data, analysis, path, params, return_metric=False, save=False, show=False):
    """Using PCA or tSNE for generated and original data visualization and returning a metric."""
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    # Prepare data
    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                            np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    # Create a string to describe the parameters for the title and file name
    param_str = f"HiddenDim={params['hidden_dim']}_Layers={params['num_layer']}_BatchSize={params['batch_size']}_Module={params['module']}"
    save_path =  f"{path}/{analysis}_{param_str}.png"

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Calculate Euclidean distance between the two sets in PCA space
        pca_distances = pairwise_distances(pca_results, pca_hat_results, metric='euclidean')
        pca_metric = np.mean(pca_distances)

        # Calculate Silhouette Score in PCA space
        all_data = np.concatenate((pca_results, pca_hat_results), axis=0)
        labels = np.array([0] * len(pca_results) + [1] * len(pca_hat_results))  # 0 for original, 1 for synthetic
        pca_silhouette = silhouette_score(all_data, labels)


        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:,0], pca_results[:,1], c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
        ax.legend()
        plt.title(f'PCA plot \n{param_str}')
        plt.xlabel('x-pca')
        plt.ylabel('y-pca')
        
        # save plot
        if save == True:
            plt.savefig(save_path)

        if show == True:
            plt.show()
        else:
            plt.close()
        
        if return_metric:
            return pca_metric, pca_silhouette

    elif analysis == 'tsne':
        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE analysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Calculate Euclidean distance between the two sets in t-SNE space
        tsne_distances = pairwise_distances(tsne_results[:anal_sample_no], tsne_results[anal_sample_no:], metric='euclidean')
        tsne_metric = np.mean(tsne_distances)

        # Calculate Silhouette Score in t-SNE space
        labels = np.array([0] * anal_sample_no + [1] * anal_sample_no)  # 0 for original, 1 for synthetic
        tsne_silhouette = silhouette_score(tsne_results, labels)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
        ax.legend()

        plt.title(f't-SNE plot \n{param_str}')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        
        # save plot
        if save == True:
            plt.savefig(save_path)
        
        if show == True:
            plt.show()
        else:
            plt.close()
            
        if return_metric:
            return tsne_metric, tsne_silhouette




def old_visualization (ori_data, generated_data, analysis, path, params):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                            np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    # Create a string to describe the parameters for the title and file name
    param_str = f"HiddenDim={params['hidden_dim']}_Layers={params['num_layer']}_BatchSize={params['batch_size']}_Module={params['module']}"
    # Save the plot with the parameters in the filename
    save_path =  f"images/Timegan/EC2/{analysis}_{param_str}.png"

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()  
        plt.title(f'PCA plot \n{param_str}')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        
        plt.savefig(save_path)

        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)
            
        # Plotting
        f, ax = plt.subplots(1)
            
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()
        
        plt.title(f't-SNE plot \n{param_str}')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(save_path)
        plt.show()    