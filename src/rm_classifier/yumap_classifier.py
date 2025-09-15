"""
Y-Shaped Parametric UMAP Classifier
===================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: MIT

"""
import sys
import os
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import parametric_umap_classifier
import helpers.yumap as yumap
import joblib
import numpy as np
import copy

class yUMAPClassifier(parametric_umap_classifier.ParametricUMAPClassifier):
    """ y-shaped Parametric UMAP Classifier with integrated clustering and outlier detection.
    This class extends the ParametricUMAPClassifier to include a classification head for semi-supervised learning.
    
    Attributes:
        classifier_head (keras.Model): The classification head to be used for semi-supervised learning. Default to None, which uses a three-layer MLP.
        umap_loss_a (float): The 'a' parameter for the UMAP loss function. Default is 1.929, which is common for UMAP with min_dist=0.
        umap_loss_b (float): The 'b' parameter for the UMAP loss function. Default is 0.7915, which is common for UMAP with min_dist=0.
        negative_sample_rate (int): The number of negative samples to use for each positive sample in the UMAP loss. Defaults to 5.
        n_classes (int): The number of classes for the classification head. Default is 100 (classes can be added or removed as needed).
        classification_loss_weight (float): The weight of the classification loss in the total loss function. Defaults to 1.0.
        kwargs: Additional keyword arguments to be passed to the ParametricUMAPClassifier.
    """
    def __init__(self, classifier_head=None,
                 umap_loss_a=1.929, # common parameters for min_dist=0
                 umap_loss_b=0.7915, # common parameters for min_dist=0
                 negative_sample_rate=5,
                 n_classes=100,
                 classification_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.classifier_head = classifier_head
        self.umap_loss_a = umap_loss_a
        self.umap_loss_b = umap_loss_b
        self.negative_sample_rate = negative_sample_rate
        self.n_classes = n_classes
        self.classification_loss_weight = classification_loss_weight

    def train_embed(self, data,
              class_labels, 
              keep_model=True, 
              save_model = None, 
              verbose=True,
              val_train_split = 0.2,
              n_training_epochs=None,
              yumap_kwargs = {},
              **kwargs
              ):
        """ Train the yUMAP model and embed the data.
        
        Args:
            data (np.ndarray): The input data to be embedded.
            class_labels (np.ndarray): The class labels for semi-supervised learning. Use -1 for unlabeled data.
            keep_model (bool): Whether to keep the trained model in the instance. Defaults to True.
            save_model (str or None): If provided, the path to save the trained model. Defauls to None.
            verbose (bool): Whether to print progress messages. Defaults to True.
            val_train_split (float): The fraction of data to use for validation during training. Defaults to 0.2.
            n_training_epochs (int or None): If provided, overrides the number of training epochs. Defaults to None.
            yumap_kwargs (dict): Additional keyword arguments to be passed to the yUMAP model. Defaults to {}.
            **kwargs: Additional keyword arguments to override instance attributes."""
        all_kwargs = {**kwargs, **{k: v for k, v in yumap_kwargs.items() if k not in kwargs}}
        new_all_kwargs = copy.deepcopy(all_kwargs)
        for key, value in new_all_kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
            if key in ['n_neighbors', 'min_dist', 'n_components', 'metric', 'repulsion_strength', 'n_training_epochs', 'loss_report_frequency', 'classifier_head', 'umap_loss_a', 'umap_loss_b', 'negative_sample_rate', 'n_classes', 'classification_loss_weight']:
                if verbose:
                    print(f"Removing {key} from kwargs.")
                all_kwargs.pop(key)
        if verbose:
            print(f"Embedding data with parameters: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, n_components={self.n_components}, metric={self.metric}")
        if n_training_epochs is not None:
            self.n_training_epochs = n_training_epochs
        yumap_model = yumap.YShapedUMAPClassifier(n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist, 
            n_components= self.n_components, 
            metric = self.metric, 
            verbose=verbose,
            repulsion_strength=self.repulsion_strength,
            n_training_epochs=self.n_training_epochs,
            loss_report_frequency=self.loss_report_frequeny,
            classifier_head=self.classifier_head,
            umap_loss_a=self.umap_loss_a,
            umap_loss_b=self.umap_loss_b,
            negative_sample_rate=self.negative_sample_rate,
            n_classes=self.n_classes,
            classification_loss_weight=self.classification_loss_weight,
            **all_kwargs
            )
        embedding = yumap_model.fit_transform(data, class_labels=class_labels, val_train_split=val_train_split)
        if save_model:
            if verbose:
                print(f"Saving UMAP model to {save_model}")
            if self.dir:
                os.sep = '/'
                os.pathsep = ':'
                save_model = os.path.join(self.dir, save_model)
            yumap_model.save(save_model)
        if keep_model:
            self.yumap_model = yumap_model
        return embedding
    
    def prepare_yumap_transform_prediction(self, data, cluster_labels=None,batch_size=20000, verbose=True):
        """ Prepare for transforming new data and predicting clusters.
        This method computes the median reconstruction loss and classification likelihood for each cluster.
        
        Args:
            data (np.ndarray): The input data to be transformed and predicted.
            cluster_labels (np.ndarray or None): If provided, the cluster labels for the input data. If None, the method will predict clusters using the current classification model.
            batch_size (int): The batch size to use for processing the data. Defaults to 20000.
            verbose (bool): Whether to print progress messages. Defaults to True.
        
        Returns:
            None
        """
        if not hasattr(self, 'yumap_model'):
            raise ValueError("Parametric UMAP model not found. Please run 'train_embed' first.")
        if verbose:
            print(f"Preparing to transform data with Parametric UMAP model and predict clusters with DBSCAN")
        reconstructions = self.reconstruct(data, batch_size=batch_size)
        rec_loss = np.mean((data - reconstructions)**2, axis=1)/data.shape[1]
        if isinstance(cluster_labels, type(None)):
            _,likelihoods,cluster_labels=self.classify(data, batch_size=batch_size, verbose=verbose)
        else:
            _,likelihoods,_=self.classify(data, batch_size=batch_size, verbose=verbose)
        likelihoods = np.max(likelihoods, axis=1)
        cluster_rec_loss, cluster_pred_likelihood = [], []
        for cluster_idx in self.cluster_indices:
            cluster_rec_loss.append(np.median(rec_loss[cluster_labels == cluster_idx]))
            cluster_pred_likelihood.append(np.median(likelihoods[cluster_labels == cluster_idx]))
        self.cluster_median_rec_loss = np.array(cluster_rec_loss)
        self.cluster_median_pred_likelihood = np.array(cluster_pred_likelihood)
        return None    
        
    def yumap_transform_prediction(self, data, batch_size=20000, verbose=True, rec_loss_noise_detection=False, rec_loss_alpha=100, rec_loss_threshold=1e6,
                                   class_loss_noise_detection=False, class_loss_alpha=0, class_loss_threshold=0):
        """ Transform new data and predict clusters with outlier detection.
        This method transforms new data using the trained yUMAP model and predicts clusters with the included classifier head.
        It also includes options for outlier detection based on reconstruction loss and classification likelihood.
        
        Args:
            data (np.ndarray): The input data to be transformed and predicted.
            batch_size (int): The batch size to use for processing the data. Defaults to 20000.
            verbose (bool): Whether to print progress messages. Defaults to True.
            rec_loss_noise_detection (bool): Whether to use reconstruction loss for outlier detection. Defaults to False.
            rec_loss_alpha (float): The multiplier for the median reconstruction loss of the predicted cluster to determine outliers. Outliers are defined as having a reconstruction loss greater than rec_loss_alpha times the median reconstruction loss of the predicted cluster. Defaults to 100.
            rec_loss_threshold (float): The absolute reconstruction loss threshold to determine outliers. Outliers are defined as having a reconstruction loss greater than this threshold. Defaults to 1e6.
            class_loss_noise_detection (bool): Whether to use classification likelihood for outlier detection. Defaults to False.
            class_loss_alpha (float): The multiplier for the median classification likelihood of the predicted cluster to determine outliers. Outliers are defined as having a classification likelihood less than class_loss_alpha times the median classification likelihood of the predicted cluster. Defaults to 0.
            class_loss_threshold (float): The absolute classification likelihood threshold to determine outliers. Outliers are defined as having a classification likelihood less than this threshold. Defaults to 0.
        
        Returns:
            predicted_cluster (np.ndarray): The predicted cluster labels for the input data. Outliers are labeled as -1.
            embedding (np.ndarray): The low-dimensional embedding of the input data.
            rec_loss (np.ndarray or None): The reconstruction loss for each input data point. Returns None if rec_loss_noise_detection is False.
            likelihoods (np.ndarray or None): The classification likelihood for each input data point. Returns None if class_loss_noise_detection is False.
        """
        
        if not hasattr(self, 'yumap_model'):
            raise ValueError("yUMAP model not found. Please run 'train_embed' first.")

        embedding, likelihoods, predicted_cluster = self.classify(data, batch_size=batch_size, verbose=verbose)
        if rec_loss_noise_detection:
            if not hasattr(self, 'cluster_median_rec_loss'):
                raise ValueError("Cluster reconstruction loss not found. Please run 'prepare_yumap_transform_prediction' first.")	
            if verbose:
                print("Calculating reconstruction loss for outlier detection")
            reconstructions = self.reconstruct(data, batch_size=batch_size)
            rec_loss = np.mean((data - reconstructions)**2, axis=1)/data.shape[1]
            predicted_cluster[rec_loss > rec_loss_threshold] = -1
            predicted_cluster[rec_loss > (self.cluster_median_rec_loss[predicted_cluster] * rec_loss_alpha)] = -1
        else:
            rec_loss = None

        if class_loss_noise_detection:
            if not hasattr(self, 'cluster_median_pred_likelihood'):
                raise ValueError("Cluster prediction likelihood not found. Please run 'prepare_yumap_transform_prediction' first.")
            if verbose:
                print("Calculating classification likelihood for outlier detection")
            likelihoods = np.max(likelihoods, axis=1)
            predicted_cluster[likelihoods < class_loss_threshold] = -1
            predicted_cluster[likelihoods < (self.cluster_median_pred_likelihood[predicted_cluster] * class_loss_alpha)] = -1
        else:
            likelihoods = None
        return predicted_cluster, embedding, rec_loss, likelihoods

    def classify(self, data, batch_size=20000, verbose=True, **kwargs):
        """ Classify new data using the trained yUMAP model.
        This method transforms new data using the trained yUMAP model and predicts clusters with the included classifier head.
        
        Args:
            data (np.ndarray): The input data to be classified.
            batch_size (int): The batch size to use for processing the data. Defaults to 20000.
            verbose (bool): Whether to print progress messages. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the yUMAP model's transform method.
        
        Returns:
            embedding (np.ndarray): The low-dimensional embedding of the input data.
            likelihoods (np.ndarray): The classification likelihood for each input data point.
            predicted_cluster (np.ndarray): The predicted cluster labels for the input data.
        """
        if not hasattr(self, 'yumap_model'):
            raise ValueError("yUMAP model not found. Please run 'train_embed' first.")
        n_batches = (len(data) + batch_size - 1) // batch_size
        embs, probs, pred_classes = np.zeros(shape=len(data), dtype=list), np.zeros(shape=len(data), dtype=list), np.zeros(shape=len(data), dtype=list)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            if verbose:
                print(f"Processing batch {start_idx} to {start_idx + batch_size}")
            batch = data[start_idx:end_idx]
            emb, prob, pred_class = self.yumap_model.transform(batch, **kwargs)
            embs[start_idx:end_idx] = emb.tolist()
            probs[start_idx:end_idx] = prob.tolist()
            pred_classes[start_idx:end_idx] = pred_class.tolist()
        pred_classes=np.array(pred_classes).astype(int)
        self.cluster_indices = np.unique(np.array(pred_classes[pred_classes != -1]))
        return np.vstack(embs), np.vstack(probs), pred_classes     
    
    def reconstruct(self, data, batch_size=20000):
        """ Reconstruct input data using the trained yUMAP model.
        This method reconstructs input data using the decoder of the trained yUMAP model.
        
        Args:
            data (np.ndarray): The input data to be reconstructed.
            batch_size (int): The batch size to use for processing the data. Defaults to 20000.
        
        Returns:
            reconstructions (np.ndarray): The reconstructed input data.
        """
        if not hasattr(self, 'yumap_model'):
            raise ValueError("yUMAP model not found. Please run 'train_embed' first.")
        n_batches = (len(data) + batch_size - 1) // batch_size
        reconstructions = np.zeros(shape=len(data), dtype=list)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            reconstructions[start_idx:end_idx] = self.yumap_model.decoder(self.yumap_model.encoder(batch)).numpy().tolist()
        return np.vstack(reconstructions)
