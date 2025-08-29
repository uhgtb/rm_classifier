"""
umap_classifier.py - a UMAP classifier especially for radio monitoring in combination with DBSCAN
================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: All Rights Reserved
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import prepare_data
from src import statistical_models, visualization
import umap.umap_ as umap
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import joblib
import copy
import warnings
import numpy as np

class UMAPClassifier:
    """
    A class to handle UMAP classification tasks especially for radio monitoring data.
    This class provides methods for data preparation, UMAP embedding, clustering with DBSCAN,
    and various prediction methods based on cluster statistics.
    """
    
    def __init__(self, 
                 yaml_path: str = None,
                 model_name: str = "umap_classifier",
                 dir: str = "umap_classifier",
                 n_neighbors: int = 35,
                 min_dist: float = 0.0,
                 n_components: int = 2,
                 metric: str = "braycurtis",
                 input_data_type: str = "time",
                 data_preparation: dict = {},
                 db_eps: float = 0.45,
                 db_min_samples: int = 50,
                 ):
        """
        Initialize the UMAP classifier with parameters.

        Args:
            yaml_path (str, optional): Path to the YAML configuration file.
                Defaults to None, which means no YAML file is loaded.
            model_name (str, optional): Name of the model.
                Defaults to "umap_classifier".
            dir (str, optional): Directory to save the model.
                Defaults to "umap_classifier".
            n_neighbors (int, optional): Number of neighbors for UMAP.
                Defaults to 35.
            min_dist (float, optional): Minimum distance between points in UMAP.
                Defaults to 0.0.
            n_components (int, optional): Number of dimensions in the latent space.
                Defaults to 2.
            metric (str, optional): Distance metric for UMAP.
                Defaults to "braycurtis".
            input_data_type (str, optional): Type of input data (e.g., "fft", "time", "fft_time", "fft_phase").
                Defaults to "time".
            data_preparation (dict, optional): Dictionary containing data preparation parameters, which are different from the default parameters.
                They include:
                - spectrum_filter: str, optional, type of filter to apply to the spectrum.
                - denoiser_n: int, number of denoiser iterations.
                - denoiser_npeak: int, number of peaks for the denoiser.
                - welch_nperseg: int, length of each segment for Welch's method.
                - welch_noverlap: int, number of points to overlap between segments.
                - welch_window: str, type of window to use in Welch's method.
                - welch_average: str, averaging method for Welch's method (mean or median).
                - windowing: str, type of windowing function to apply.
                - log_filter: bool, whether to apply a logarithmic filter.
                - cut_beacon_frequencies: bool, whether to cut beacon frequencies.
                - beacon_frequencies: list, frequencies of the beacons.
                - beacon_width: float, width of the beacons.
                Defaults to an empty dictionary.
            db_eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other in DBSCAN.
                Defaults to 0.45.
            db_min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point in DBSCAN. 
                Defaults to 50.
        
        """
        self.yaml_path = yaml_path
        self.model_name = model_name
        self.dir = dir
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.input_data_type = input_data_type
        self.data_preparation = data_preparation
        self.db_eps = db_eps
        self.db_min_samples = db_min_samples
        if yaml_path:
            self._load_yaml(yaml_path)
        default_data_preparation_dict = {
            "target_data_type": "fft",
            "suppress_dc": True,
            "spectrum_filter": None, # None, welch, denoiser
            "windowing": None, # None, hamming, hanning, blackman, blackmanharris, bartlett, kaiser
            "log_filter": False,
            "cut_beacon_frequencies": False,
            "normalization": False, # standard zscore normalization
            "avg_pooling": None, # int of an precedent avg_pooling layer,
            "max_pooling": None, # int of an precedent max_pooling layer,
            "denoiser_n": 20,
            "denoiser_npeak": 3,
            "welch_nperseg": 512,
            "welch_noverlap": 0,
            "welch_window": 'boxcar',
            "welch_average": 'mean',
            "beacon_frequencies": None,
            "beacon_width": None,
            "frequency_bins": None,
            "sampling_frequency": 180,  # Default sampling frequency in MHz
        }
        self.data_preparation = {**default_data_preparation_dict, **self.data_preparation}
            
    def _load_yaml(self, yaml_path: str):
        """
        Load parameters from a YAML file.
        
        Args:
            yaml_path (str): Path to the YAML file.
        """
        import yaml
        with open(yaml_path, 'r') as file:
            params = yaml.safe_load(file)
            for key, value in params.items():
                setattr(self, key, value)
    
    def prepare_data(self, data, verbose=True, data_preparation: dict = {},**kwargs):
        """
        Prepare data for UMAP classification.

        Args:
            data (np.ndarray): Input data to be prepared.
            verbose (bool, optional): Whether to print preparation details. Defaults to True.
            data_preparation (dict, optional): Dictionary containing data preparation parameters to override the class defaults
                Defaults to an empty dictionary.
            **kwargs: Additional keyword arguments to override class attributes.

        Returns:
            np.ndarray: Prepared data ready for UMAP classification.
        """
        for key, value in kwargs.items():
            if key in self.data_preparation:
                if verbose:
                    print(f"Overriding {key} in data_preparation dict with value {value} from kwargs.")
                self.data_preparation[key] = value
            elif key != "data_preparation":
                if verbose:
                    print(f"Adding new key {key} with value {value} to the class instance.")
                self.__setattr__(key, value)
        for key, value in data_preparation.items():
            if verbose:
                print(f"Overriding {key} with value {value} from data_preparation dict.")
            self.data_preparation[key] = value

        if verbose:
            print(f"Preparing data with parameters: {self.data_preparation}")
        prepared_data = prepare_data.prepare_data(self, data, verbose)
        del data
        prepared_data, self.data_preparation = prepare_data.normalize_data(self, prepared_data, verbose)
        prepared_data = prepare_data.pooling_data(self, prepared_data, verbose)
        return prepared_data
    
    def custom_prepare_data(self, data, custom_prepare_fcn = None, verbose=True, **kwargs):
        """
        Prepare data using a custom preparation function.

        Args:
            data (np.ndarray): Input data to be prepared.
            custom_prepare_fcn (callable): Custom function to prepare the data. It should accept
                the data as its first argument and return the prepared data.
            verbose (bool, optional): Whether to print preparation details. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the custom preparation function.
        Returns:
            np.ndarray: Prepared data.  
        """
        if custom_prepare_fcn is None:
            raise ValueError("custom_prepare_fcn must be provided.")
        if verbose:
            print(f"Preparing data with custom function: {custom_prepare_fcn.__name__}")
        self.data_preparation["custom_prepare_fcn"] = custom_prepare_fcn
        return custom_prepare_fcn(data, verbose=verbose, **kwargs)
    
    def embed(self, data, keep_model=True, save_model = None, verbose=True, umap_kwargs={}, **kwargs):
        """
        Embed data into a lower-dimensional space using UMAP.

        Args:
            data (np.ndarray): Input data to be embedded.
            keep_model (bool, optional): Whether to save the trained UMAP model as the class variable self.umap_model. Defaults to True.
            save_model (str, optional): Path to save the trained UMAP model using joblib. Defaults to None, which means the model is not saved to disk.
            verbose (bool, optional): Whether to print embedding details. Defaults to True.
            umap_kwargs (dict, optional): Additional keyword arguments to pass to the UMAP constructor. Defaults to empty dict.
            **kwargs: Additional keyword arguments to override class attributes.

        Returns:
            np.ndarray: Embedded data in lower-dimensional space.
            """
        
        for key, value in kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
        if verbose:
            print(f"Embedding data with parameters: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, n_components={self.n_components}, metric={self.metric}")
        
        umap_model = umap.UMAP(n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist, 
            n_components= self.n_components, 
            metric = self.metric, 
            verbose=verbose,
            **umap_kwargs
            ).fit(data)
        embedding = umap_model.transform(data)
        if save_model:
            if verbose:
                print(f"Saving UMAP model to {save_model}")
            with open(save_model, 'wb') as f:
                joblib.dump(umap_model, f)
        if keep_model:
            self.umap_model = umap_model
        return embedding
    
    def _set_cluster_indices(self, labels):
        labels = labels[labels != -1]  # Exclude noise points
        self.cluster_indices = np.unique(labels)
        return self.cluster_indices
    
    def _dbscan(self, data, keep_model=True, save_model = None, verbose=True, **kwargs):
        db_model = DBSCAN(eps=self.db_eps, min_samples=self.db_min_samples).fit(data)
        if save_model:
            if verbose:
                print(f"Saving DBSCAN model to {save_model}")
            with open(save_model, 'wb') as f:
                joblib.dump(db_model, f)
        if keep_model:
            self.db_model = db_model
        self._set_cluster_indices(db_model.labels_)
        return db_model.labels_
    
    def _hdbscan(self, data, keep_model=True, save_model = None, verbose=True, **hdb_kwargs):
        hdb_model = HDBSCAN(min_samples=self.hdb_min_samples, **hdb_kwargs).fit(data)
        if save_model:
            if verbose:
                print(f"Saving HDBSCAN model to {save_model}")
            with open(save_model, 'wb') as f:
                joblib.dump(hdb_model, f)
        if keep_model:
            self.hdb_model = hdb_model
        self._set_cluster_indices(hdb_model.labels_)
        return hdb_model.labels_
    
    def db_classify(self, data, keep_model=True, save_model = None, automatic_parameters = False, verbose=True, **kwargs):
        """
        Classify data using DBSCAN clustering.

        Args:
            data (np.ndarray): Input data to be classified.
            keep_model (bool, optional): Whether to save the trained DBSCAN model as the class variable self.db_model. Defaults to True.
            save_model (str, optional): Path to save the trained DBSCAN model using joblib. Defaults to None, which means the model is not saved to disk.
            automatic_parameters (bool, optional): Whether to automatically determine optimal DBSCAN parameters. Defaults to False.
            verbose (bool, optional): Whether to print classification details. Defaults to True.
            **kwargs: Additional keyword arguments to override class attributes.
            
        Returns:
            np.ndarray: Cluster labels for each data point."""
        for key, value in kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
        if automatic_parameters:
            pass #TODO: implement automatic parameter selection
        else:
            return self._dbscan(data, keep_model=keep_model, save_model=save_model, verbose=verbose, **kwargs)
        
    def hdb_classify(self, data, keep_model=True, save_model = None, verbose=True, hdb_min_samples =5, **kwargs):
        """
        Classify data using HDBSCAN clustering.

        Args:
            data (np.ndarray): Input data to be classified.
            keep_model (bool, optional): Whether to save the trained HDBSCAN model as the class variable self.hdb_model. Defaults to True.
            save_model (str, optional): Path to save the trained HDBSCAN model using joblib. Defaults to None, which means the model is not saved to disk.
            verbose (bool, optional): Whether to print classification details. Defaults to True.
            hdb_min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point in HDBSCAN. 
                Defaults to 5.
            **kwargs: Additional keyword arguments to override class attributes.
        Returns:
            np.ndarray: Cluster labels for each data point."""
        for key, value in kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
        self.hdb_min_samples = hdb_min_samples
        return self._hdbscan(data, keep_model=keep_model, save_model=save_model, verbose=verbose, **kwargs)
    

    ### Cluster trace methods
    def save_cluster_trace(self, data, clusters, trace_statistic="mean", verbose=True, modifier_function_name="", q=0.1):
        """
        Save cluster as class variables traces based on the provided statistic.
        Args:
            data (np.ndarray): The input data from which to calculate cluster traces.
            clusters (np.ndarray): Array of cluster indices for each data point.
            trace_statistic (str, optional): Statistic to use for calculating cluster traces. 
                Options are "mean", "median", "min", "max", "std", or "quantile". Defaults to "mean".
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            q (float, optional): Quantile value to use if trace_statistic is "quantile". Must be between 0 and 1. Defaults to 0.1.
        """
        if np.all(self._set_cluster_indices(clusters) != self.cluster_indices):
            warnings.warn("Provided clusters do not match the model's cluster indices. Setting them to the model's cluster indices.")
            self._set_cluster_indices(clusters)
        if len(clusters) != len(data):
            raise ValueError("Length of cluster indices must match length of data.")
        cluster_traces = []
        for cluster_idx in self.cluster_indices:
            if trace_statistic == "mean":
                cluster_trace = np.mean(data[clusters == cluster_idx], axis=0)
            elif trace_statistic == "median":
                cluster_trace = np.median(data[clusters == cluster_idx], axis=0)
            elif trace_statistic == "std":
                cluster_trace = np.std(data[clusters == cluster_idx], axis=0)
            elif trace_statistic == "max":
                cluster_trace = np.max(data[clusters == cluster_idx], axis=0)
            elif trace_statistic == "min":
                cluster_trace = np.min(data[clusters == cluster_idx], axis=0)
            elif trace_statistic == "quantile":
                try:
                    cluster_trace = np.quantile(data[clusters == cluster_idx], q, axis=0)
                except ValueError:
                    raise ValueError("trace_statistic must start with 'quantile_' and followed by a float value.")
            else:
                raise ValueError("Invalid cluster_statistic. Choose from 'mean', 'median', 'min', 'max', 'std' or 'quantile'.")
            cluster_traces.append(cluster_trace)
        if trace_statistic == "quantile":
            if hasattr(self, f'cluster_{modifier_function_name}quantile_traces'):
                getattr(self,f'cluster_{modifier_function_name}quantile_traces')[q] = np.array(cluster_traces)
            else:
                self.__setattr__(f'cluster_{modifier_function_name}quantile_traces', {q: np.array(cluster_traces)})
        else:
            self.__setattr__(f'cluster_{modifier_function_name}{trace_statistic}_traces', np.array(cluster_traces))
        if verbose:
            print(f"Saved {len(cluster_traces)} cluster traces with {trace_statistic} statistic.")
        del cluster_traces

    def get_cluster_trace(self, cluster_idx, trace_statistic="mean", modifier_function_name="", q=0.5):
        """
        Retrieve the trace for a specific cluster based on the provided statistic.
        Args:   
            cluster_idx (int): The index of the cluster for which to retrieve the trace.
            trace_statistic (str, optional): Statistic used for calculating the cluster trace.
                Options are "mean", "median", "min", "max", "std", or "quantile". Defaults to "mean".
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            q (float, optional): Quantile value to use if trace_statistic is "quantile". Must be between 0 and 1. Defaults to 0.5.
        Returns:
            np.ndarray: The trace of the specified cluster.
        """
        if trace_statistic == "quantile":
            quantile_traces = getattr(self, f'cluster_{modifier_function_name}quantile_traces', {})
            return quantile_traces.get(q, None)[self.cluster_indices == cluster_idx][0]
        else:
            return getattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces')[self.cluster_indices == cluster_idx][0]
    
    def save_modified_cluster_trace(self, data, clusters, modifier_function, modifier_function_name=None, trace_statistic="mean", verbose=True):
        """
        Apply a modifier function to the data and save the cluster traces based on the provided statistic.
        Args:
            data (np.ndarray): The input data from which to calculate cluster traces.
            clusters (np.ndarray): Array of cluster indices for each data point.
            modifier_function (callable): Function to modify the data before calculating traces.
            modifier_function_name (str, optional): Name of the modifier function. If None, the function's __name__ attribute is used.
            trace_statistic (str, optional): Statistic to use for calculating cluster traces.
                Options are "mean", "median", "min", "max", "std", or "quantile". Defaults to "mean".
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        """
        if modifier_function_name is None:
            modifier_function_name = modifier_function.__name__
        data = modifier_function(data)       
        self.save_cluster_trace(data, clusters, trace_statistic=trace_statistic, verbose=verbose, modifier_function_name=modifier_function_name)

    def save_cluster_cholesky_factors(self, data, clusters, eps=1e-8, verbose=True, modifier_function_name=""):
        """
        Save cluster Cholesky factors as class variables.       
        Args:
            data (np.ndarray): The input data from which to calculate cluster Cholesky factors.
            clusters (np.ndarray): Array of cluster indices for each data point.
            eps (float, optional): Small value added to the diagonal of the covariance matrix for numerical stability. Defaults to 1e-8.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating Cholesky factors.
                This is used to differentiate between factors calculated on raw and modified data. Defaults to an empty string.
        """
        if np.all(self._set_cluster_indices(clusters) != self.cluster_indices):
            warnings.warn("Provided clusters do not match the model's cluster indices. Setting them to the model's cluster indices.")
            self._set_cluster_indices(clusters)
        if len(clusters) != len(data):
            raise ValueError("Length of cluster indices must match length of data.")
        cluster_W_matrices, cluster_log_dets = [], []
        for cluster_idx in self.cluster_indices:
            cov_matrix = np.cov(data[clusters == cluster_idx], rowvar=False)
            W, log_det = statistical_models.cholesky_factors(cov_matrix, eps=eps)
            cluster_W_matrices.append(W)
            cluster_log_dets.append(log_det)
        self.__setattr__(f'cluster_{modifier_function_name}W_matrices_traces', np.array(cluster_W_matrices))
        self.__setattr__(f'cluster_{modifier_function_name}log_dets_traces', np.array(cluster_log_dets))
        if verbose:
            print(f"Saved {len(cluster_W_matrices)} cluster Cholesky factors.")
        del cluster_W_matrices, cluster_log_dets

    def save_modified_cluster_cholesky_factors(self, data, clusters, modifier_function, modifier_function_name=None, eps=1e-8, verbose=True):
        """
        Apply a modifier function to the data and save the cluster Cholesky factors.
        Args:
            data (np.ndarray): The input data from which to calculate cluster Cholesky factors.
            clusters (np.ndarray): Array of cluster indices for each data point.
            modifier_function (callable): Function to modify the data before calculating Cholesky factors.
            modifier_function_name (str, optional): Name of the modifier function. If None, the function's __name__ attribute is used.
            eps (float, optional): Small value added to the diagonal of the covariance matrix for numerical stability. Defaults to 1e-8.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        """
        if modifier_function_name is None:
            modifier_function_name = modifier_function.__name__
        data = modifier_function(data)        
        self.save_cluster_cholesky_factors(data, clusters, eps=eps, verbose=verbose, modifier_function_name=modifier_function_name)

    ### Prediction methods
    def prepare_minimum_distance_prediction(self, data, clusters, metric="euclidean", trace_statistic="mean", modifier_function_name="", verbose=True):
        """
        Prepare for minimum distance prediction by calculating and saving the median distances of each cluster trace.
        And also ensures that the cluster traces are calculated and saved.
        Args:
            data (np.ndarray): The input data to be classified.
            clusters (np.ndarray): Array of cluster indices for each data point.
            metric (str, optional): Distance metric to use. Defaults to 'euclidean'.
            trace_statistic (str, optional): Statistic used for calculating the cluster traces.
                Options are "mean", "median", "min", "max", "std", or "quantile". Defaults to "mean".
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:
            None
                """
        if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
            self.save_cluster_trace(data, clusters, trace_statistic=trace_statistic, verbose=verbose, modifier_function_name=modifier_function_name)
        compared_traces = getattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces')
        median_cluster_distance = []
        for cluster_idx in self.cluster_indices:
            cluster_data = data[clusters == cluster_idx]
            median_cluster_distance.append(np.median(cdist(cluster_data, [compared_traces[cluster_idx]], metric=metric)))
        median_cluster_distance = np.array(median_cluster_distance)/data.shape[1]  # Normalize by number of features
        setattr(self, f'cluster_{modifier_function_name}{trace_statistic}_median_distances_{metric}', np.array(median_cluster_distance))
        return None

    def minimum_distance_prediction(self, data, metric="euclidean", max_distance=1.0, noise_alpha=100, trace_statistic="mean", modifier_function_name="", verbose=True):
        """
        Predict cluster labels based on the minimum distance to cluster traces.
        Args:
            data (np.ndarray): The input data to be classified.
            metric (str, optional): Distance metric to use. Defaults to "euclidean".
            noise_alpha (float, optional): Multiplier for the median distance to determine noise threshold.
                Points with a distance greater than median_distance * noise_alpha are labeled as noise (-1). Defaults to 100.
            max_distance (float, optional): Maximum distance to consider a point as belonging to a cluster.
                Points with a distance greater than max_distance are labeled as noise (-1). Defaults to 1.0.
            trace_statistic (str, optional): Statistic used for calculating the cluster traces.
                Options are "mean", "median", "min", "max", "std", or "quantile". Defaults to "mean".
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:
            np.ndarray: Predicted cluster labels for each data point.
            np.ndarray: Minimum distances to the nearest cluster trace for each data point.
        """
        if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
            raise ValueError(f"Cluster traces with statistic '{trace_statistic}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_modified_cluster_trace' first.")
        
        distances = np.array(cdist(data,getattr(self,f'cluster_{modifier_function_name}{trace_statistic}_traces'), metric=metric))/data.shape[1]  # Normalize by number of features
        predicted_cluster = self.cluster_indices[np.argmin(distances, axis=1)]
        minimum_distances = np.min(distances, axis=1)
        predicted_cluster[minimum_distances > max_distance] = -1  # Mark as noise if distance exceeds max_distance
        predicted_cluster[minimum_distances > (getattr(self, f'cluster_{modifier_function_name}{trace_statistic}_median_distances_{metric}')[predicted_cluster] * noise_alpha)] = -1  # Mark as noise if distance exceeds median distance * noise_alpha
        return predicted_cluster, minimum_distances
    
    def prepare_ml_uncorrelated_normal_prediction(self, data, clusters, modifier_function_name="", verbose=True):
        """
        Prepare for maximum likelihood prediction assuming uncorrelated normal distributions by calculating and saving the median uncorrelated normal likelihoods of each cluster.
        And also ensures that the cluster mean and std traces are calculated and saved.
        Args:
            data (np.ndarray): The input data to be classified.
            clusters (np.ndarray): Array of cluster indices for each data point.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:
            None
        """
        if not hasattr(self, f'cluster_{modifier_function_name}mean_traces'):
            self.save_cluster_trace(data, clusters, trace_statistic="mean", verbose=verbose, modifier_function_name=modifier_function_name)
        if not hasattr(self, f'cluster_{modifier_function_name}std_traces'):
            self.save_cluster_trace(data, clusters, trace_statistic="std", verbose=verbose, modifier_function_name=modifier_function_name)
        if verbose:
            print("Calculating median uncorrelated normal likelihoods for each cluster.") 
        median_logls=[]
        bin_means = getattr(self, f'cluster_{modifier_function_name}mean_traces')
        bin_stds = getattr(self, f'cluster_{modifier_function_name}std_traces')
        for cluster_idx in self.cluster_indices:
            
            cluster_data = data[clusters == cluster_idx].copy()
            median_logls.append(np.median(statistical_models.logl_uncorrelated_normal(cluster_data, bin_means[cluster_idx:cluster_idx+1], bin_stds[cluster_idx:cluster_idx+1])))
        median_logls = np.array(median_logls)
        setattr(self, f'cluster_{modifier_function_name}_median_uncorrelated_normal_likelihood', np.exp(median_logls))
        return None     

    def ml_uncorrelated_normal_prediction(self, data, noise_alpha=0, min_logl=-100, modifier_function_name="",
                                            data_batch_size=1000, cluster_batch_size=30, verbose=True):
        """
        Predict cluster labels based on maximum likelihood estimation assuming uncorrelated normal distributions.
        Args:
            data (np.ndarray): The input data to be classified.
            noise_alpha (float, optional): Multiplier for the median likelihood to determine noise threshold.
                Points with a likelihood below median_likelihood * noise_alpha are labeled as noise (-1). Defaults to 0.
            min_logl (float, optional): Minimum log-likelihood threshold to consider a point as belonging to a cluster.
                Points with a log-likelihood below min_logl are labeled as noise (-1). Defaults to -100.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            data_batch_size (int, optional): Size of data batches for processing. Defaults to 1000.
            cluster_batch_size (int, optional): Size of cluster batches for processing. Defaults to 30.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:
            np.ndarray: Predicted cluster labels for each data point.
            np.ndarray: Maximum log-likelihood values for each data point.
        """
        for trace_statistic in ["mean", "std"]:
            if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
                raise ValueError(f"Cluster traces with statistic '{trace_statistic}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_modified_cluster_trace' first.")
        bin_means = getattr(self, f'cluster_{modifier_function_name}mean_traces')
        bin_stds = getattr(self, f'cluster_{modifier_function_name}std_traces')
        max_logl, predicted_cluster = [], []
        for j in range(0, len(data), data_batch_size):
            logl = []
            jmax = np.min((j+data_batch_size, len(data)))
            data_batch = data[j:jmax]
            for i in range(0, len(self.cluster_indices), cluster_batch_size):
                imax = np.min((i+cluster_batch_size, len(self.cluster_indices)))
                bin_means_batch, bin_stds_batch = bin_means[i:imax], bin_stds[i:imax]
                logl.append(statistical_models.logl_uncorrelated_normal(data_batch, bin_means_batch, bin_stds_batch))
            logl = np.array(np.concatenate(logl, axis=0))
            predicted_cluster.append(self.cluster_indices[np.argmax(logl, axis=0)])
            max_logl.append(np.max(logl, axis=0))

        predicted_cluster = np.array(np.concatenate(predicted_cluster, axis=0))
        max_logl = np.array(np.concatenate(max_logl, axis=0)) 
        predicted_cluster[max_logl < min_logl] = -1  # Mark as noise if log-likelihood is below threshold
        predicted_cluster[np.exp(max_logl) < (getattr(self, f'cluster_{modifier_function_name}_median_uncorrelated_normal_likelihood')[predicted_cluster] * noise_alpha)] = -1  # Mark as noise if likelihood is below median likelihood * noise_alpha
        return predicted_cluster, max_logl
    
    def prepare_ml_correlated_normal_prediction(self, data, clusters, modifier_function_name="", verbose=True):
        """
        Prepare for maximum likelihood prediction assuming correlated normal distributions by calculating and saving the Cholesky factors of each cluster.
        And also ensures that the cluster mean traces are calculated and saved.
        Args:
            data (np.ndarray): The input data to be classified.
            clusters (np.ndarray): Array of cluster indices for each data point.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:
            None
        """
        if not hasattr(self, f'cluster_{modifier_function_name}mean_traces'):
            self.save_cluster_trace(data, clusters, trace_statistic="mean", verbose=verbose, modifier_function_name=modifier_function_name)
        if not hasattr(self, f'cluster_{modifier_function_name}W_matrices_traces') or not hasattr(self, f'cluster_{modifier_function_name}log_dets_traces'):
            self.save_cluster_cholesky_factors(data, clusters, verbose=verbose, modifier_function_name=modifier_function_name)
        if verbose:
            print("Calculating median correlated normal likelihoods for each cluster.") 
        median_logls=[]
        bin_means = getattr(self, f'cluster_{modifier_function_name}mean_traces')
        W_matrices = getattr(self, f'cluster_{modifier_function_name}W_matrices_traces')
        log_dets = getattr(self, f'cluster_{modifier_function_name}log_dets_traces')
        
        for cluster_idx in self.cluster_indices:  
            cluster_data = data[clusters == cluster_idx].copy()
            median_logls.append(np.median(statistical_models.logl_normal(cluster_data, 
                                                                         bin_means[cluster_idx:cluster_idx+1], 
                                                                         W_matrices[cluster_idx:cluster_idx+1], 
                                                                         log_dets[cluster_idx:cluster_idx+1])))
        median_logls = np.array(median_logls)  # Normalize likelihood by number of features

        setattr(self, f'cluster_{modifier_function_name}_median_correlated_normal_likelihood', np.exp(median_logls))
        return None  
    
    def ml_correlated_normal_prediction(self, data, min_logl=-100, noise_alpha=0, modifier_function_name="",
                                            data_batch_size=1000, cluster_batch_size=30, verbose=True):
        """
        Predict cluster labels based on maximum likelihood estimation assuming correlated normal distributions.
        Args:
            data (np.ndarray): The input data to be classified.
            min_logl (float, optional): Minimum log-likelihood threshold to consider a point as belonging to a cluster.
                Points with a log-likelihood below min_logl are labeled as noise (-1). Defaults to -100.
            noise_alpha (float, optional): Multiplier for the median likelihood to determine noise threshold.
                Points with a likelihood below median_likelihood * noise_alpha are labeled as noise (-1). Defaults to 0.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before calculating traces.
                This is used to differentiate between traces calculated on raw and modified data. Defaults to an empty string.
            data_batch_size (int, optional): Size of data batches for processing. Defaults to 1000.
            cluster_batch_size (int, optional): Size of cluster batches for processing. Defaults to 30.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:
            np.ndarray: Predicted cluster labels for each data point.
            np.ndarray: Maximum log-likelihood values for each data point.
        """
        # make sure the cluster properties are available
        for trace_statistic in ["mean", "W_matrices", "log_dets"]:
            if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
                raise ValueError(f"Cluster traces with statistic '{trace_statistic}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_cluster_cholesky_factors' first.")

        bin_means = getattr(self, f'cluster_{modifier_function_name}mean_traces')
        W_matrices = getattr(self, f'cluster_{modifier_function_name}W_matrices_traces')
        log_dets = getattr(self, f'cluster_{modifier_function_name}log_dets_traces')
        max_logl, predicted_cluster = [], []

        for j in range(0, len(data), data_batch_size):
            logl = []
            jmax = np.min((j+data_batch_size, len(data)))
            data_batch = data[j:jmax]
            for i in range(0, len(self.cluster_indices), cluster_batch_size):
                imax = np.min((i+cluster_batch_size, len(self.cluster_indices)))
                bin_means_batch, W_matrices_batch, log_dets_batch = bin_means[i:imax], W_matrices[i:imax], log_dets[i:imax]
                logl.append(statistical_models.logl_normal(data_batch, bin_means_batch, W_matrices_batch, log_dets_batch))
            logl = np.array(np.concatenate(logl, axis=0))
            predicted_cluster.append(self.cluster_indices[np.argmax(logl, axis=0)])
            max_logl.append(np.max(logl, axis=0))
            
        predicted_cluster = np.array(np.concatenate(predicted_cluster, axis=0))
        max_logl = np.array(np.concatenate(max_logl, axis=0))
        predicted_cluster[max_logl < min_logl] = -1  # Mark as noise if log-likelihood is below threshold
        predicted_cluster[np.exp(max_logl) < (getattr(self, f'cluster_{modifier_function_name}_median_correlated_normal_likelihood')[predicted_cluster] * noise_alpha)] = -1  # Mark as noise if likelihood is below median likelihood * noise_alpha
        return predicted_cluster, max_logl
    
    def _dbscan_predict(self, model, X, data_batch_size=20000):
            """Predict clusters for new data points based on trained DBSCAN model.
            """
            
            # For very large datasets, process in batches
            nr_samples = X.shape[0]
            y_new = np.ones(shape=nr_samples, dtype=int) * -1
            
            # Build nearest neighbors model on DBSCAN components for fast lookup
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(model.components_)
            
            # Process in batches to avoid memory issues
            for start_idx in range(0, nr_samples, data_batch_size):
                end_idx = min(start_idx + data_batch_size, nr_samples)
                batch = np.array(X[start_idx:end_idx])
                
                # Find distance to closest component for each point in batch
                distances, indices = nbrs.kneighbors(batch)
                
                # Get the corresponding cluster labels for points within eps
                mask = distances.ravel() < model.eps
                y_new[start_idx:end_idx][mask] = model.labels_[
                    model.core_sample_indices_[indices.ravel()[mask]]]
            return y_new
    
    def umap_transform_prediction(self, data, db_eps=None, data_batch_size=20000, verbose=True):
        """
        Transform data using the trained UMAP model and predict clusters using the trained DBSCAN model.
        Args:
            data (np.ndarray): The input data to be transformed and classified.
            db_eps (float, optional): Epsilon parameter for DBSCAN. If None, uses the value from the trained DBSCAN model. Defaults to None.
            data_batch_size (int, optional): Size of data batches for processing. Defaults to 20000.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        Returns:        
            np.ndarray: Predicted cluster labels for each data point.
            np.ndarray: UMAP embeddings for each data point.
        """
        if not hasattr(self, 'umap_model'):
            raise ValueError("UMAP model not found. Please run 'embed' first.")
        if not hasattr(self, 'db_model'):
            raise ValueError("DBSCAN model not found. Please run 'classify' first.")
        if verbose:
            print(f"Transforming data with UMAP model and predicting clusters with DBSCAN (eps={db_eps})")

        n_samples = data.shape[0]
        embedding = np.zeros(shape=n_samples, dtype=list)
        n_batches = (len(data) + data_batch_size - 1) // data_batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * data_batch_size
            end_idx = min((batch_idx + 1) * data_batch_size, len(data))
            if verbose:
                print(f"Processing batch {start_idx} to {start_idx + data_batch_size}")
            batch = data[start_idx:end_idx]
            embedding[start_idx:end_idx] = self.umap_model.transform(batch).tolist()
        if verbose:
            print("Creating DBSCAN model")
        if not isinstance(db_eps, type(None)):
            dbscan_model = copy.deepcopy(self.db_model)
            dbscan_model.eps = db_eps
        else:
            dbscan_model = self.db_model
        predictions=self._dbscan_predict(dbscan_model, np.vstack(embedding), data_batch_size=data_batch_size)

        return predictions, np.vstack(embedding)
    
    ### Visualization methods
    def plot_spectra(self, k, modifier_function_name="", sigma=2, q=None, **kwargs):
        """
        Plot the spectra of the cluster traces.
        Args:
            k (int): The index of the cluster to plot.
            modifier_function_name (str, optional): Name of the modifier function applied to the data before plotting the traces.
                This is used to differentiate between traces plotted and the raw data. Defaults to an empty string.
            sigma (float, optional): Number of standard deviations to plot around the mean trace. If None, standard deviation is not plotted. Defaults to 2.
            q (float, optional): Quantile value to plot if quantile traces are available. Must be between 0 and 1. If None, quantile traces are not plotted. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plotting function.
                These can include:
                - plot_type (list, str): Type of input data to plot. Options are 'fft', 'time', 'phase' or 'default' which are all provided input data types.
                - save_fig (str): Path to save the figure. If None, the figure is not saved. Defaults to None.
        Returns:
            fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
        """
        if sigma:
            for trace_statistic in ["mean", "std"]:
                if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
                    raise ValueError(f"Cluster traces with statistic '{trace_statistic}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_cluster_cholesky_factors' first.")
        if q:
            for trace_statistic in ["quantile","median"]:
                if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
                    raise ValueError(f"Cluster traces with statistic '{trace_statistic}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_cluster_cholesky_factors' first.")
                if not q in getattr(self, f'cluster_{modifier_function_name}quantile_traces'):
                    raise ValueError(f"Cluster traces with quantile '{q}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_cluster_cholesky_factors' first.")
                if not 1-q in getattr(self, f'cluster_{modifier_function_name}quantile_traces'):
                    raise ValueError(f"Cluster traces with quantile '{1-q}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_cluster_cholesky_factors' first.")
        return visualization.plot_spectra(self, k, modifier_function_name=modifier_function_name, sigma=sigma, q=q, **kwargs)
    
    def plot_cluster_samples(self, prepared_data, clusters, cluster_idx, **kwargs):
        """
        Plot samples from a specific cluster.
        Args:
            prepared_data (np.ndarray): The prepared data used for clustering.
            clusters (np.ndarray): Array of cluster indices for each data point.
            cluster_idx (int): The index of the cluster to plot samples from.
            **kwargs: Additional keyword arguments to pass to the plotting function.
                These can include:
                - n_samples (int): Number of samples to plot. Defaults to 100.
                - random_state (int): Random seed for reproducibility. Defaults to None.
                - plot_type (list, str): Type of input data to plot. Options are 'fft', 'time', 'phase' or 'default' which are all provided input data types.
                - save_fig (str): Path to save the figure. If None, the figure is not saved. Defaults to None.
        Returns:
            fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
        """
        return visualization.plot_cluster_samples(self, prepared_data, clusters, cluster_idx, **kwargs)
    
    def plot_index(self, prepared_data, index, **kwargs):
        """
        Plot data samples based on a provided index array.
        Args:
            prepared_data (np.ndarray): The prepared data used for clustering.
            index (np.ndarray): Array of indices to plot. Should be the same length as prepared_data.
            **kwargs: Additional keyword arguments to pass to the plotting function.
                These can include:
                - plot_type (list, str): Type of input data to plot. Options are 'fft', 'time', 'phase' or 'default' which are all provided input data types.
                - save_fig (str): Path to save the figure. If None, the figure is not saved. Defaults to None.
        Returns:
            fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True"""
        return visualization.plot_index(self, prepared_data, index, **kwargs)
    
    def plot_embedding(self, embedding, labels=None, **kwargs):
        """
        Plot the UMAP embedding with optional labels.
        Args:
            embedding (np.ndarray): The UMAP embedding to plot.
            labels (np.ndarray, optional): Array of labels for coloring the embedding. If None, no labels are used. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plotting function.
                These can include:
                - save_fig (str): Path to save the figure. If None, the figure is not saved. Defaults to None.
                - label_name (str): Name of the labels for the colorbar. Defaults to "Cluster".
                - label_type (str): Type of labels. Options are 'categorical' or 'continuous'. Defaults to 'categorical'.
                - alpha (float): Alpha value for point transparency. Defaults to 0.01.
        Returns:
            fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
                """
        return visualization.plot_embdding(embedding, labels, **kwargs)
    
    def plot_overview_bokeh(self, embedding, color_key=None, title="UMAP Overview", save_fig=None, **hover_kwargs):
        """
        Create an interactive Bokeh plot of the UMAP embedding.
        Args:
            embedding (np.ndarray): The UMAP embedding to plot.
            color_key (str or None): Attribute name in hover_kwargs to use for coloring the points. If None, no coloring is applied.
            title (str, optional): Title of the plot. Defaults to "UMAP Overview".
            save_fig (str, optional): Path to save the Bokeh plot as an HTML file
            **hover_kwargs: Additional keyword arguments to pass to the plotting function essentially arrays with the same length as embedding, which contain information about each point in the embedding, which is displayed when hovering over the points.
        """
        return visualization.plot_overview_bokeh(embedding, color_key=color_key, title=title, save_fig=save_fig, **hover_kwargs)