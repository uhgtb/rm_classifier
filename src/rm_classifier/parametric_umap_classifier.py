"""
Parametric UMAP Classifier Extension of the UMAPClassifier
==========================================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: MIT

"""
import sys
import os
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import umap_classifier
import rm_classifier.helpers.tracked_parametric_umap as tracked_parametric_umap
import numpy as np
import copy

class ParametricUMAPClassifier(umap_classifier.UMAPClassifier):
    """ Subclass of UMAPClassifier that uses Parametric UMAP for embedding and DBSCAN for clustering.
    Furthermore, an autoencoder allows for reconstruction of the input data, which can be used for outlier detection.
    
    Attributes:
        repulsion_strength (float): The strength of the repulsion term in UMAP. Defaults to 1.0.
        n_training_epochs (int): Number of training epochs for the Parametric UMAP model. Defaults to 100.
        loss_report_frequeny (int): Frequency of loss reporting during a training epoch. Defaults to 1000. Since the number of learnable relations is the square of the datapoints, this should be set to a high value, if large datasets are processed.
        kwargs: Additional keyword arguments for the UMAPClassifier base class.
    """
    
    def __init__(self, repulsion_strength = 1.0,
                 n_training_epochs=100,
                 loss_report_frequeny=1000, 
                 **kwargs):
        super().__init__(**kwargs)
        self.parametric_model = None
        self.repulsion_strength = repulsion_strength
        self.n_training_epochs = n_training_epochs
        self.loss_report_frequeny = loss_report_frequeny

    def train_embed(self, data, 
              keep_model=True, 
              save_model = None, 
              verbose=True,
              val_train_split = 0.2,
              n_training_epochs=None,
              pumap_kwargs = {},
              **kwargs
              ):
        """ Train the Parametric UMAP model on the given data and return the embedding.
        
        Args:
            data (array-like): The input data to be embedded.
            keep_model (bool): Whether to keep the trained model in the instance. Defaults to True.
            save_model (str or None): Path to save the trained model. If None, the model is not saved. Defaults to None.
            verbose (bool): Whether to print verbose output during training. Defaults to True.
            val_train_split (float): Fraction of data to use for validation during training. Defaults to 0.2.
            n_training_epochs (int or None): Number of training epochs for the Parametric UMAP model. If None, uses the instance's n_training_epochs attribute. Defaults to None.
            pumap_kwargs (dict): Additional keyword arguments for the Parametric UMAP model.
            **kwargs: Additional keyword arguments that can override instance attributes.
        
        Returns:
            embedding (array-like): The resulting embedding of the input data.
        """
        all_kwargs = {**kwargs, **{k: v for k, v in pumap_kwargs.items() if k not in kwargs}}
        new_all_kwargs = copy.deepcopy(all_kwargs)
        for key, value in new_all_kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
            if key in ['n_neighbors', 'min_dist', 'n_components', 'metric', 'repulsion_strength', 'n_training_epochs', 'loss_report_frequency']:
                if verbose:
                    print(f"Removing {key} from kwargs.")
                all_kwargs.pop(key)
        if verbose:
            print(f"Embedding data with parameters: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, n_components={self.n_components}, metric={self.metric}")
        if n_training_epochs is not None:
            self.n_training_epochs = n_training_epochs

        pumap_model = tracked_parametric_umap.TrackedPUMAP(
            n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist, 
            n_components= self.n_components, 
            metric = self.metric, 
            verbose=verbose,
            repulsion_strength=self.repulsion_strength,
            n_training_epochs=self.n_training_epochs,
            loss_report_frequency=self.loss_report_frequeny,
            **all_kwargs
            )
        embedding = pumap_model.fit_transform(data, val_train_split=val_train_split)
        if save_model:
            if verbose:
                print(f"Saving UMAP model to {save_model}")
            if self.dir:
                save_model = os.path.join(self.dir, save_model)
            pumap_model.save(save_model)
        if keep_model:
            self.pumap_model = pumap_model
        return embedding
    
    def prepare_pumap_transform_prediction(self, data, cluster_labels,data_batch_size=20000, verbose=True):
        """ Prepare the Parametric UMAP model for transforming new data and predicting clusters.
        This method calculates the median reconstruction loss for each cluster, which can be used for outlier detection during prediction.
        
        Args:
            data (array-like): The input data to be used for calculating reconstruction loss.
            cluster_labels (array-like): The cluster labels corresponding to the input data.
            data_batch_size (int): The batch size to use for processing the data. Defaults to 20000.
            verbose (bool): Whether to print verbose output during processing. Defaults to True.
        
        Returns:
            None
        """
        if not hasattr(self, 'pumap_model'):
            raise ValueError("Parametric UMAP model not found. Please run 'train_embed' first.")
        if not hasattr(self, 'db_model'):
            raise ValueError("DBSCAN model not found. Please run 'classify' first.")
        if verbose:
            print(f"Preparing to transform data with Parametric UMAP model and predict clusters with DBSCAN")
        reconstructions = self.reconstruct(data, batch_size=data_batch_size)
        rec_loss = np.mean((data - reconstructions)**2, axis=1)/data.shape[1]
        cluster_rec_loss = []
        for cluster_idx in self.cluster_indices:
            cluster_rec_loss.append(np.median(rec_loss[cluster_labels == cluster_idx]))
        self.cluster_median_rec_loss = np.array(cluster_rec_loss)
        return None
        
    
    def _create_pumap_transform_prediction(self, data, db_eps=None, data_batch_size=20000, verbose=True):

        n_samples = data.shape[0]
        embedding = np.zeros(shape=n_samples, dtype=list)
        n_batches = (len(data) + data_batch_size - 1) // data_batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * data_batch_size
            end_idx = min((batch_idx + 1) * data_batch_size, len(data))
            if verbose:
                print(f"Processing batch {start_idx} to {start_idx + data_batch_size}")
            batch = data[start_idx:end_idx]
            embedding[start_idx:end_idx] = self.pumap_model.transform(batch).tolist()
        if verbose:
            print("Creating DBSCAN model")
        if not isinstance(db_eps, type(None)):
            dbscan_model = copy.deepcopy(self.db_model)
            dbscan_model.eps = db_eps
        else:
            dbscan_model = self.db_model
        predictions=self._dbscan_predict(dbscan_model, np.vstack(embedding), data_batch_size=data_batch_size)

        return predictions, np.vstack(embedding)
    
        
    def pumap_transform_prediction(self, data, db_eps=None, data_batch_size=20000, verbose=True, rec_loss_noise_detection=False, noise_alpha=100, rec_loss_threshold=1e6):
        """ Transform new data using the trained Parametric UMAP model and predict clusters using the trained DBSCAN model.
        Optionally, use reconstruction loss for outlier detection.
        
        Args:
            data (array-like): The input data to be transformed and clustered.
            db_eps (float or None): The epsilon parameter for the DBSCAN model. If None, uses the epsilon from the trained model. Defaults to None.
            data_batch_size (int): The batch size to use for processing the data. Defaults to 20000.
            verbose (bool): Whether to print verbose output during processing. Defaults to True.
            rec_loss_noise_detection (bool): Whether to use reconstruction loss for outlier detection. Defaults to False.
            noise_alpha (float): The multiplier for the median reconstruction loss to determine outliers as points with reconstruction loss greater than noise_alpha times the median reconstruction loss of their assigned cluster. Defaults to 100.
            rec_loss_threshold (float): A hard threshold for reconstruction loss to determine outliers. Points with reconstruction loss greater than this value are marked as outliers. Defaults to 1e6.
        
        Returns:
            predicted_cluster (array-like): The predicted cluster labels for the input data.
            embedding (array-like): The embedding of the input data.
            rec_loss (array-like or None): The reconstruction loss for each point in the input data. Returns None if rec_loss_noise_detection is False.
        """
        if not hasattr(self, 'pumap_model'):
            raise ValueError("UMAP model not found. Please run 'train_embed' first.")
        if not hasattr(self, 'db_model'):
            raise ValueError("DBSCAN model not found. Please run 'classify' first.")
        if verbose:
            print(f"Transforming data with UMAP model and predicting clusters with DBSCAN (eps={db_eps})")

        predicted_cluster, embedding = self._create_pumap_transform_prediction(data, db_eps=db_eps, data_batch_size=data_batch_size, verbose=verbose)
        if rec_loss_noise_detection:
            if not hasattr(self, 'cluster_median_rec_loss'):
                raise ValueError("Cluster reconstruction loss not found. Please run 'prepare_pumap_transform_prediction' first.")	
            if verbose:
                print("Calculating reconstruction loss for outlier detection")
            reconstructions = self.reconstruct(data, batch_size=data_batch_size)
            rec_loss = np.mean((data - reconstructions)**2, axis=1)/data.shape[1]
            predicted_cluster[rec_loss > rec_loss_threshold] = -1
            predicted_cluster[rec_loss > (self.cluster_median_rec_loss[predicted_cluster] * noise_alpha)] = -1
        
        if rec_loss_noise_detection:
            return predicted_cluster, embedding, rec_loss
        else:
            return predicted_cluster, embedding, None

    
    def reconstruct(self, data, batch_size=20000):
        """ Reconstruct the input data using the autoencoder part of the Parametric UMAP model.
        
        Args:
            data (array-like): The input data to be reconstructed.
            batch_size (int): The batch size to use for processing the data. Defaults to 20000.
        
        Returns:
            reconstructions (array-like): The reconstructed data.
        """
        if not hasattr(self, 'pumap_model'):
            raise ValueError("parametric UMAP model not found. Please run 'train_embed' first.")
        n_batches = (len(data) + batch_size - 1) // batch_size
        reconstructions = np.zeros(shape=len(data), dtype=list)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            reconstructions[start_idx:end_idx] = self.pumap_model.decoder(self.pumap_model.encoder(batch)).numpy().tolist()
        return np.vstack(reconstructions)
