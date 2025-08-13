#import custom modules
import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import prepare_data
import statistical_models
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import joblib
import warnings



#import standard libraries
import numpy as np

class UMAPClassifier:
    """
    A class to handle UMAP classification tasks.
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
        
        Parameters:
        -----------
        yaml_path : str, optional
            Path to the YAML configuration file.
        model_name : str, optional
            Name of the model.
        dir : str, optional
            Directory to save the model.
        n_neighbors : int, optional
            Number of neighbors for UMAP.
        min_dist : float, optional
            Minimum distance between points in UMAP.
        n_components : int, optional
            Number of dimensions for UMAP output.
        metric : str, optional
            Distance metric for UMAP.
        input_data_type : str, optional
            Type of input data (e.g., "fft", "time", "fft_time", "fft_phase").
        data_preparation : dict, optional
            Dictionary containing data preparation parameters, which are different from the default parameters.
            They include:
            - suppress_zero: bool, whether to suppress zero values in the data.
            - spectrum_filter: str, optional, type of filter to apply to the spectrum.
            - denoiser_n: int, number of denoiser iterations.
            - denoiser_npeak: int, number of peaks for the denoiser.
            - welch_nperseg: int, length of each segment for Welch's method.
            - welch_noverlap: int, number of points to overlap between segments.
            - welch_window: str, type of window to use in Welch's method.
            - welch_average: str, averaging method for Welch's method.
            - windowing: str, type of windowing function to apply.
            - log_filter: bool, whether to apply a logarithmic filter.
            - cut_beacon_frequencies: bool, whether to cut beacon frequencies.
            - beacon_frequencies: list, frequencies of the beacons.
            - beacon_width: float, width of the beacons.
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
        
        Parameters:
        -----------
        yaml_path : str
            Path to the YAML file containing parameters.
        """
        import yaml
        with open(yaml_path, 'r') as file:
            params = yaml.safe_load(file)
            for key, value in params.items():
                setattr(self, key, value)
    
    def prepare_data(self, data, verbose=True, data_preparation: dict = {},**kwargs):
        """
        Prepare data for UMAP classification.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data to be prepared.
        verbose : bool, optional
            Whether to print preparation details.
        
        Returns:
        --------
        prepared_data : np.ndarray
            Prepared data ready for UMAP classification.
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
        if custom_prepare_fcn is None:
            raise ValueError("custom_prepare_fcn must be provided.")
        if verbose:
            print(f"Preparing data with custom function: {custom_prepare_fcn.__name__}")
        self.data_preparation["custom_prepare_fcn"] = custom_prepare_fcn
        return custom_prepare_fcn(data, verbose=verbose, **kwargs)
    
    def embed(self, data, keep_model=True, save_model = None, verbose=True, **kwargs):
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
            verbose=0,
            angular_rp_forest=True,
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
    
    def classify(self, data, keep_model=True, save_model = None, automatic_parameters = False, verbose=True, **kwargs):
        for key, value in kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
        if automatic_parameters:
            pass #TODO: implement automatic parameter selection
        else:
            return self._dbscan(data, keep_model=keep_model, save_model=save_model, verbose=verbose, **kwargs)
    
    def save_cluster_trace(self, data, clusters, trace_statistic="mean", verbose=True, modifier_function_name=""):
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
            else:
                raise ValueError("Invalid cluster_statistic. Choose from 'mean', 'median', 'min', 'max' or 'std'.")
            cluster_traces.append(cluster_trace)
        self.__setattr__(f'cluster_{modifier_function_name}{trace_statistic}_traces', np.array(cluster_traces))
        if verbose:
            print(f"Saved {len(cluster_traces)} cluster traces with {trace_statistic} statistic.")
        del cluster_traces
    
    def save_modified_cluster_trace(self, data, clusters, modifier_function, modifier_function_name=None, trace_statistic="mean", verbose=True):
        if modifier_function_name is None:
            modifier_function_name = modifier_function.__name__
        data = modifier_function(data)
        self.save_cluster_trace(data, clusters, trace_statistic=trace_statistic, verbose=verbose, modifier_function_name=modifier_function_name)

    def minimum_distance_prediction(self, data, metric="euclidean", max_distance=1.0, trace_statistic="mean", modifier_function_name="", verbose=True):
        if not hasattr(self, f'cluster_{modifier_function_name}{trace_statistic}_traces'):
            raise ValueError(f"Cluster traces with statistic '{trace_statistic}' and modifier function '{modifier_function_name}' not found. Please run 'save_cluster_trace' or 'save_modified_cluster_trace' first.")
        
        distances = cdist(data,getattr(self,f'cluster_{modifier_function_name}{trace_statistic}_traces'), metric=metric)
        predicted_cluster = self.cluster_indices[np.argmin(distances, axis=1)]
        minimum_distances = np.min(distances, axis=1)
        predicted_cluster[minimum_distances > max_distance] = -1  # Mark as noise if distance exceeds max_distance
        return predicted_cluster, minimum_distances
    
    def ml_uncorrelated_normal_prediction(self, data, min_logl=-1, modifier_function_name="",
                                          noise_limit=0.1, data_batch_size=1000, cluster_batch_size=100, verbose=True):
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
        return predicted_cluster, max_logl