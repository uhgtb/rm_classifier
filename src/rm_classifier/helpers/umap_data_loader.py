"""
Functions to Load Data 
======================
Loads data in batches, perform UMAP embedding and clustering on each batch, and select a high-entropy subset of the data.

Author: Johann Luca Kastner
Date: 15/09/2025
License: MIT
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
import os
from rm_classifier.helpers import high_entropy_subset
import joblib
import warnings
import numpy as np
import gc


def load_umap_data_from_dir(umap_classifier, dir_path, data_loader = None, data_loader_kwargs = {}, 
                            max_umap_events = 1e5, batch_size = None, n_selected_per_batch = None, summary_batch_size = None,
                            local_embedding = True, return_raw = False,
                            umap_kwargs = {}, clustering_kwargs = {}, clustering_mode = "dbscan",
                            file_extension = ".pkl", verbose=True):
    """ Load data from a directory, seperate into batches, perform UMAP embedding and clustering on each batch, and select a high-entropy subset of the data.
    
    Args:
        umap_classifier (umap_classifier.UMAPClassifier): An instance of the UMAP classifier.
        dir_path (str): Path to the directory containing data files. 
        data_loader (callable): A function that takes a file path and returns a tuple (data, properties). Here data is a numpy array of shape (n_samples, n_features) and properties is an optional numpy array of shape (n_samples, n_properties). Default is None, which uses a default loader that expects .pkl files containing pandas DataFrames.
        data_loader_kwargs (dict): Additional keyword arguments to pass to the data_loader function. Defaults to {}.
        max_umap_events (int): Maximum number of events to select for UMAP embedding. Default is 1e5.
        batch_size (int): Number of events to process in each batch. If None defaults to max_umap_events. Defaults to None.
        n_selected_per_batch (int): Number of events to select from each batch. If None defaults to max_umap_events/10. Defaults to None.
        summary_batch_size (int): Number of events to accumulate before performing a final selection. If None defaults to batch_size. Defaults to None.
        local_embedding (bool): Whether to use parameters for UMAP and DBSCAN supporting local embedding and clustering for each batch. Default is True.
        return_raw (bool): Whether to return the raw data along with the processed data. Default is False.
        umap_kwargs (dict): Additional keyword arguments to pass to the UMAP embedding function. Defaults to {}.
        clustering_kwargs (dict): Additional keyword arguments to pass to the clustering function. Defaults to {}.
        clustering_mode (str): Clustering algorithm to use, either "dbscan" or "hdbscan". Default is "dbscan".
        file_extension (str): File extension of the data files to load. Default is ".pkl".
        verbose (bool): Whether to print progress messages. Default is True.     
        
    Returns:
        selected_data (np.ndarray): A numpy array of shape (n_selected_events, n_features) containing the selected events after UMAP embedding and clustering.
        selected_raw_data (np.ndarray): A numpy array of shape (n_selected_events, n_raw_features) containing the raw data of the selected events. Returned only if return_raw is True.
        selected_properties (np.ndarray): A numpy array of shape (n_selected_events, n_properties) containing the properties of the selected events. Returned only if properties are provided by the data_loader.
    """

    if local_embedding:
        if "n_neighbors" not in umap_kwargs.keys():
            umap_kwargs["n_neighbors"] = 5
        if clustering_mode == "dbscan":
            if "eps" not in clustering_kwargs.keys():
                clustering_kwargs["eps"] = 0.1
                clustering_kwargs["min_samples"] = 5
        if clustering_mode == "hdbscan":
            if "min_cluster_size" not in clustering_kwargs.keys():
                clustering_kwargs["min_cluster_size"] = 5
            if "min_samples" not in clustering_kwargs.keys():
                clustering_kwargs["min_samples"] = 5
    restore_original = {}
    for key in umap_kwargs.keys():
        if hasattr(umap_classifier, key):
            restore_original[key] = getattr(umap_classifier, key)
            setattr(umap_classifier, key, umap_kwargs[key])
            if verbose:
                print(f"Temporarily overriding {key} with value {umap_kwargs[key]} from umap_kwargs.")
    for key in clustering_kwargs.keys():
        if hasattr(umap_classifier, key):
            restore_original[key] = getattr(umap_classifier, key)
            setattr(umap_classifier, key, clustering_kwargs[key])
            if verbose:
                print(f"Temporarily overriding {key} with value {clustering_kwargs[key]} from clustering_kwargs.")

    memmap_flag = True
    def default_data_loader(file_path):
        df = joblib.load(file_path)
        return np.stack(df[umap_classifier.input_data_type].to_numpy()), None
    max_umap_events = int(max_umap_events)
    if isinstance(batch_size, type(None)):
        batch_size = max_umap_events
    if isinstance(n_selected_per_batch, type(None)):
        n_selected_per_batch = max_umap_events/10
    if isinstance(summary_batch_size, type(None)):
        summary_batch_size = batch_size
    batch_size = int(batch_size)
    n_selected_per_batch = int(n_selected_per_batch)
    summary_batch_size = int(summary_batch_size)

    
    if data_loader is None:
        data_loader = default_data_loader
    current_length, current_selected_length = 0, 0
    for file_path in Path(dir_path).rglob('*'):
        if file_path.suffix == file_extension:
            if verbose:
                print(f"Loading data from {file_path}...")
            data, properties = data_loader(file_path, **data_loader_kwargs)
            data_length = data.shape[1]
            n_batches = int(np.ceil(data.shape[0]/batch_size))
            for i in range(n_batches):
                if verbose:
                    print(f"Processing batch {i+1}/{n_batches}...")
                i_min = i*batch_size
                i_max = min((i+1)*batch_size, data.shape[0])
                data_batch = data[i_min:i_max]
                properties_batch = None if properties is None else properties[i_min:i_max]
                prepared_data = umap_classifier.prepare_data(data_batch, verbose=verbose)
                del data_batch
                embeddings=umap_classifier.train_embed(prepared_data, **umap_kwargs, verbose=verbose)
                if verbose:
                    print("Embeddings shape:", embeddings.shape)
                if clustering_mode == "dbscan":
                    db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                elif clustering_mode == "hdbscan":
                    db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                else:
                    raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                if verbose:
                    print("DBSCAN clusters found:", set(db_clusters))
                batch_selected_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, min(n_selected_per_batch, i_max-i_min))
                
                if memmap_flag:
                    os.sep = '/'
                    os.pathsep = ':'
                    os.makedirs(os.path.join(umap_classifier.dir, "umap_loaded_data"), exist_ok=True)
                    umap_classifier.umap_data_dir = os.path.join(umap_classifier.dir, "umap_loaded_data")
                    current_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "current_data.dat"), dtype=np.float32, mode='w+', shape=(summary_batch_size,prepared_data.shape[1]))
                    selected_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "selected_data.dat"), dtype=np.float32, mode='w+', shape=(max_umap_events,prepared_data.shape[1]))
                    if properties is not None:
                        current_properties = np.memmap(os.path.join(umap_classifier.umap_data_dir, "current_properties.dat"), dtype=object, mode='w+', shape=(summary_batch_size,properties.shape[1]))
                        selected_properties = np.memmap(os.path.join(umap_classifier.umap_data_dir, "selected_properties.dat"), dtype=object, mode='w+', shape=(max_umap_events,properties.shape[1]))
                    if return_raw:
                        current_raw_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "current_raw_data.dat"), dtype=np.float32, mode='w+', shape=(summary_batch_size,data_length))
                        selected_raw_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "selected_raw_data.dat"), dtype=np.float32, mode='w+', shape=(max_umap_events,data_length))
                    memmap_flag = False

                current_data[current_length:current_length + sum(batch_selected_mask)] = prepared_data[batch_selected_mask]
                current_data.flush()
                if properties is not None:
                    current_properties[current_length:current_length + sum(batch_selected_mask)] = properties_batch[batch_selected_mask]
                    current_properties.flush()
                if return_raw:
                    current_raw_data[current_length:current_length + sum(batch_selected_mask)] = data[i_min:i_max][batch_selected_mask]
                    current_raw_data.flush()
                    print("saved raw data")
                current_length += sum(batch_selected_mask)
                if verbose:
                    print(f"Selected {sum(batch_selected_mask)} events, current total {current_length}/{summary_batch_size}.")

                del prepared_data
                del embeddings
                del db_clusters
                del batch_selected_mask

                # Find clusters in current selection
                if current_length > summary_batch_size-n_selected_per_batch:
                    if verbose:
                        print(f"Reached {current_length} events, performing final selection...")
                    if current_selected_length > 0:
                        embeddings=umap_classifier.train_embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
                        if clustering_mode == "dbscan":
                            selected_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        elif clustering_mode == "hdbscan":
                            selected_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        else:
                            raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                        del embeddings
                    else:
                        selected_db_clusters = np.array([])

                    # Find clusters in current batch
                    embeddings=umap_classifier.train_embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
                    if clustering_mode == "dbscan":
                        batch_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    elif clustering_mode == "hdbscan":
                        batch_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    else:
                        raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                    del embeddings

                    # Determine how many to select from batch vs previous selection
                    batch_db_clusters[batch_db_clusters == 0] = -1
                    batch_db_clusters[batch_db_clusters != -1] *= -1 # Invert cluster labels to avoid overlaps
                    complete_mask = high_entropy_subset.high_entropy_subset_mask(np.concatenate(
                        [selected_db_clusters,batch_db_clusters]), min(current_selected_length + current_length, max_umap_events))
                    del batch_db_clusters, selected_db_clusters

                    if current_selected_length > 0:
                        selected_mask = complete_mask[:current_selected_length]
                        n_selected = sum(selected_mask)
                    else:
                        n_selected = 0
                    batch_mask = complete_mask[current_selected_length:current_selected_length+current_length]
                    n_batch = sum(batch_mask)
                    if verbose:
                        print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")

                    # Merge selections
                    if current_selected_length > 0:
                        selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
                        if return_raw:
                            selected_raw_data[:n_selected] = selected_raw_data[:current_selected_length][selected_mask]
                        if properties is not None:
                            selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
                    selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
                    selected_data.flush()
                    if return_raw:
                        selected_raw_data[n_selected:n_selected+n_batch] = current_raw_data[:current_length][batch_mask]
                        selected_raw_data.flush()
                    if properties is not None:
                        selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
                        selected_properties.flush()
                    current_selected_length = n_selected + n_batch
                    current_length = 0
                    current_data[:] = 0 
                    current_data.flush()
        gc.collect()                       
    # Finalize selection
    if verbose:
        print(f"Reached the end of data loading, performing final selection...")
    if current_length > 0:
            # Find clusters in current selection
        if current_selected_length > 0:
            embeddings=umap_classifier.train_embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
            if clustering_mode == "dbscan":
                selected_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
            elif clustering_mode == "hdbscan":
                selected_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
            else:
                raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
            del embeddings
        else:
            selected_db_clusters = np.array([])

        # Find clusters in current batch
        embeddings=umap_classifier.train_embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
        if clustering_mode == "dbscan":
            batch_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
        elif clustering_mode == "hdbscan":
            batch_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
        else:
            raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
        del embeddings

        # Determine how many to select from batch vs previous selection
        batch_db_clusters[batch_db_clusters == 0] = -1
        batch_db_clusters[batch_db_clusters != -1] *= -1 # Invert cluster labels to avoid overlaps
        complete_mask = high_entropy_subset.high_entropy_subset_mask(np.concatenate(
            [selected_db_clusters,batch_db_clusters]), min(current_selected_length + current_length, max_umap_events))
        del batch_db_clusters, selected_db_clusters

        if current_selected_length > 0:
            selected_mask = complete_mask[:current_selected_length]
            n_selected = sum(selected_mask)
        else:
            n_selected = 0
        batch_mask = complete_mask[current_selected_length:current_selected_length+current_length]
        n_batch = sum(batch_mask)
        if verbose:
            print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")

        # Merge selections
        if current_selected_length > 0:
            selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
            if properties is not None:
                selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
            if return_raw:
                selected_raw_data[:n_selected] = selected_raw_data[:current_selected_length][selected_mask]
        selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
        if properties is not None:
            selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
        if return_raw:
            selected_raw_data[n_selected:n_selected+n_batch] = current_raw_data[:current_length][batch_mask]
        selected_data.flush()
        selected_raw_data.flush() if return_raw else None
        selected_properties.flush() if properties is not None else None
        current_data_filepath = current_data.filename
        current_data._mmap.close()
        os.unlink(current_data_filepath)
        if properties is not None:
            current_properties_filepath = current_properties.filename
            current_properties._mmap.close()
            os.unlink(current_properties_filepath)
        if return_raw:
            current_raw_data_filepath = current_raw_data.filename
            current_raw_data._mmap.close()
            os.unlink(current_raw_data_filepath)

    for key in restore_original.keys():
        setattr(umap_classifier, key, restore_original[key])
        if verbose:
            print(f"Restored original {key} with value {restore_original[key]}.")                
    if return_raw:
        return selected_data[:current_selected_length], selected_raw_data[:current_selected_length], selected_properties[:current_selected_length] if properties is not None else None
    else:
        return selected_data[:current_selected_length], None, selected_properties[:current_selected_length] if properties is not None else None    

            
    


def load_umap_data_batchwise_from_dir(umap_classifier, dir_path, data_batch_loader_kwargs = {}, data_batch_loader = None,
                            max_umap_events = 1e5, batch_size = None, n_selected_per_batch = None, summary_batch_size = None,
                            local_embedding = True, return_raw = False,
                            umap_kwargs = {}, clustering_kwargs = {}, clustering_mode = "dbscan",
                            file_extension = ".pkl", verbose=True):
    """ Load data from a directory in batches, perform UMAP embedding and clustering on each batch, and select a high-entropy subset of the data.
    
    Args:
        umap_classifier (umap_classifier.UMAPClassifier): An instance of the UMAP classifier.
        dir_path (str): Path to the directory containing data files.
        data_batch_loader (callable): A function that takes a batch index and a file path, and returns a tuple (data_batch, properties_batch, change_file). Here data_batch is a numpy array of shape (n_samples_in_batch, n_features), properties_batch is an optional numpy array of shape (n_samples_in_batch, n_properties), and change_file is a boolean indicating whether to move to the next file. Default is None, which uses a default loader that expects .pkl files containing pandas DataFrames.
        data_batch_loader_kwargs (dict): Additional keyword arguments to pass to the data_batch_loader function. Defaults to {}.
        max_umap_events (int): Maximum number of events to select for UMAP embedding. Defaults to 1e5.
        batch_size (int): Number of events to process in each batch. If None defaults to max_umap_events. Defaults to None.
        n_selected_per_batch (int): Number of events to select from each batch. If None defaults to max_umap_events/10. Defaults to None.
        summary_batch_size (int): Number of events to accumulate before performing a final selection. If None defaults to batch_size. Defaults to None.
        local_embedding (bool): Whether to use parameters for UMAP and DBSCAN supporting local embedding and clustering for each batch. Default is True.
        return_raw (bool): Whether to return the raw data along with the processed data. Default is False.
        umap_kwargs (dict): Additional keyword arguments to pass to the UMAP embedding function. Defaults to {}.
        clustering_kwargs (dict): Additional keyword arguments to pass to the clustering function. Defaults to {}.
        clustering_mode (str): Clustering algorithm to use, either "dbscan" or "hdbscan". Default is "dbscan".
        file_extension (str): File extension of the data files to load. Default is ".pkl".
        verbose (bool): Whether to print progress messages. Default is True.
    
    Returns:
        selected_data (np.ndarray): A numpy array of shape (n_selected_events, n_features) containing the selected events after UMAP embedding and clustering.
        selected_raw_data (np.ndarray): A numpy array of shape (n_selected_events, n_raw_features) containing the raw data of the selected events. Returned only if return_raw is True.
        selected_properties (np.ndarray): A numpy array of shape (n_selected_events, n_properties) containing the properties of the selected events. Returned only if properties are provided by the data_batch_loader.
    """
    
    if local_embedding:
        if "n_neighbors" not in umap_kwargs.keys():
            umap_kwargs["n_neighbors"] = 5
        if clustering_mode == "dbscan":
            if "eps" not in clustering_kwargs.keys():
                clustering_kwargs["eps"] = 0.1
                clustering_kwargs["min_samples"] = 5
        if clustering_mode == "hdbscan":
            if "min_cluster_size" not in clustering_kwargs.keys():
                clustering_kwargs["min_cluster_size"] = 5
            if "min_samples" not in clustering_kwargs.keys():
                clustering_kwargs["min_samples"] = 5
    restore_original = {}
    for key in umap_kwargs.keys():
        if hasattr(umap_classifier, key):
            restore_original[key] = getattr(umap_classifier, key)
            setattr(umap_classifier, key, umap_kwargs[key])
            if verbose:
                print(f"Temporarily overriding {key} with value {umap_kwargs[key]} from umap_kwargs.")
    for key in clustering_kwargs.keys():
        if hasattr(umap_classifier, key):
            restore_original[key] = getattr(umap_classifier, key)
            setattr(umap_classifier, key, clustering_kwargs[key])
            if verbose:
                print(f"Temporarily overriding {key} with value {clustering_kwargs[key]} from clustering_kwargs.")
    memmap_flag = True
    max_umap_events = int(max_umap_events)
    if isinstance(batch_size, type(None)):
        batch_size = max_umap_events
    if isinstance(n_selected_per_batch, type(None)):
        n_selected_per_batch = max_umap_events/10
    if isinstance(summary_batch_size, type(None)):
        summary_batch_size = batch_size
    batch_size = int(batch_size)
    n_selected_per_batch = int(n_selected_per_batch)
    summary_batch_size = int(summary_batch_size)
    def default_data_batch_loader(batch_idx,file_path):
        df = joblib.load(file_path)
        i_min = batch_idx*batch_size
        i_max = min((batch_idx+1)*batch_size, len(df))
        if i_max >= len(df):
            return np.stack(df[umap_classifier.input_data_type].to_numpy())[i_min:i_max], None, True
        else:
            return np.stack(df[umap_classifier.input_data_type].to_numpy())[i_min:i_max], None, False
    
    if data_batch_loader is None:
        data_batch_loader = default_data_batch_loader
    current_length, current_selected_length = 0, 0
    for file_path in Path(dir_path).rglob('*'):
        if file_path.suffix == file_extension:
            if verbose:
                print(f"Loading data from {file_path}...")
            batch_idx = -1
            change_file = False
            while not change_file:
                batch_idx += 1
                data_batch, properties_batch, change_file = data_batch_loader(batch_idx, file_path, **data_batch_loader_kwargs)
                if verbose:
                    print(f"Processing batch {batch_idx+1}...")
                if len(data_batch) > batch_size:
                    warnings.warn(f"Batch size {len(data_batch)} of data is larger than specified batch_size {batch_size}. Cutting down batch.")
                    data_batch = data_batch[:batch_size]
                if properties_batch is not None and len(properties_batch) > batch_size:
                    warnings.warn(f"Batch size {len(properties_batch)} of properties is larger than specified batch_size {batch_size}. Cutting down batch.")
                    properties_batch = properties_batch[:batch_size]
                
                prepared_data = umap_classifier.prepare_data(data_batch, verbose=verbose)
                if not return_raw:
                    del data_batch
                embeddings=umap_classifier.train_embed(prepared_data, **umap_kwargs, verbose=verbose)
                if clustering_mode == "dbscan":
                    db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                elif clustering_mode == "hdbscan":
                    db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                else:
                    raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                batch_selected_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, min(n_selected_per_batch, len(db_clusters)))
                
                if memmap_flag:
                    original_sep = os.sep
                    original_pathsep = os.pathsep
                    os.sep = '/'
                    os.pathsep = ':'
                    os.makedirs(os.path.join(umap_classifier.dir, "umap_loaded_data"), exist_ok=True)
                    umap_classifier.umap_data_dir = os.path.join(umap_classifier.dir, "umap_loaded_data")
                    current_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "current_data.dat"), dtype=np.float32, mode='w+', shape=(summary_batch_size,prepared_data.shape[1]))
                    selected_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "selected_data.dat"), dtype=np.float32, mode='w+', shape=(max_umap_events,prepared_data.shape[1]))
                    if properties_batch is not None:
                        selected_properties = np.memmap(os.path.join(umap_classifier.umap_data_dir, "selected_properties.dat"), dtype=object, mode='w+', shape=(max_umap_events,properties_batch.shape[1]))
                        current_properties = np.memmap(os.path.join(umap_classifier.umap_data_dir, "current_properties.dat"), dtype=object, mode='w+', shape=(summary_batch_size,properties_batch.shape[1]))
                    if return_raw:
                        current_raw_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "current_raw_data.dat"), dtype=np.float32, mode='w+', shape=(summary_batch_size,data_batch.shape[1]))
                        selected_raw_data = np.memmap(os.path.join(umap_classifier.umap_data_dir, "selected_raw_data.dat"), dtype=np.float32, mode='w+', shape=(max_umap_events,data_batch.shape[1]))
                    memmap_flag = False
                    os.sep = original_sep
                    os.pathsep = original_pathsep

                current_data[current_length:current_length + sum(batch_selected_mask)] = prepared_data[batch_selected_mask]
                if properties_batch is not None:
                    current_properties[current_length:current_length + sum(batch_selected_mask)] = properties_batch[batch_selected_mask]
                if return_raw:
                    current_raw_data[current_length:current_length + sum(batch_selected_mask)] = data_batch[batch_selected_mask]
                    del data_batch
                current_length += sum(batch_selected_mask)
                if verbose:
                    print(f"Selected {sum(batch_selected_mask)} events, current total {current_length}/{summary_batch_size}.")

                del prepared_data
                del embeddings
                del db_clusters
                del batch_selected_mask

                # Find clusters in current selection
                if current_length > summary_batch_size-n_selected_per_batch:
                    if verbose:
                        print(f"Reached {current_length} events, performing final selection...")
                    if current_selected_length > 0:
                        embeddings=umap_classifier.train_embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
                        if clustering_mode == "dbscan":
                            selected_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        elif clustering_mode == "hdbscan":
                            selected_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        else:
                            raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                        del embeddings
                    else:
                        selected_db_clusters = np.array([])

                    # Find clusters in current batch
                    embeddings=umap_classifier.train_embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
                    if clustering_mode == "dbscan":
                        batch_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    elif clustering_mode == "hdbscan":
                        batch_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    else:
                        raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                    del embeddings

                    # Determine how many to select from batch vs previous selection
                    batch_db_clusters[batch_db_clusters == 0] = -1
                    batch_db_clusters[batch_db_clusters != -1] *= -1 # Invert cluster labels to avoid overlaps
                    complete_mask = high_entropy_subset.high_entropy_subset_mask(np.concatenate(
                        [selected_db_clusters,batch_db_clusters]), min(current_selected_length + current_length, max_umap_events))
                    del batch_db_clusters, selected_db_clusters

                    if current_selected_length > 0:
                        selected_mask = complete_mask[:current_selected_length]
                        n_selected = sum(selected_mask)
                    else:
                        n_selected = 0
                    batch_mask = complete_mask[current_selected_length:current_selected_length+current_length]
                    n_batch = sum(batch_mask)
                    if verbose:
                        print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")

                    # Merge selections
                    if current_selected_length > 0:
                        selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
                        if properties_batch is not None:
                            selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
                        if return_raw:
                            selected_raw_data[:n_selected] = selected_raw_data[:current_selected_length][selected_mask]
                    selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
                    if return_raw:
                        selected_raw_data[n_selected:n_selected+n_batch] = current_raw_data[:current_length][batch_mask]
                    if properties_batch is not None:
                        selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
                    selected_data.flush()
                    current_selected_length = n_selected + n_batch
                    current_length = 0
                    current_data[:] = 0   
        gc.collect()                     
    # Finalize selection
    if verbose:
        print(f"Reached the end of data loading, performing final selection...")
    if current_length > 0:
            # Find clusters in current selection
        if current_selected_length > 0:
            embeddings=umap_classifier.train_embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
            if clustering_mode == "dbscan":
                selected_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
            elif clustering_mode == "hdbscan":
                selected_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
            else:
                raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
            del embeddings
        else:
            selected_db_clusters = np.array([])

        # Find clusters in current batch
        embeddings=umap_classifier.train_embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
        if clustering_mode == "dbscan":
            batch_db_clusters=umap_classifier.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
        elif clustering_mode == "hdbscan":
            batch_db_clusters=umap_classifier.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
        else:
            raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
        del embeddings
        
       # Determine how many to select from batch vs previous selection
        batch_db_clusters[batch_db_clusters == 0] = -1
        batch_db_clusters[batch_db_clusters != -1] *= -1 # Invert cluster labels to avoid overlaps
        complete_mask = high_entropy_subset.high_entropy_subset_mask(np.concatenate(
            [selected_db_clusters,batch_db_clusters]), min(current_selected_length + current_length, max_umap_events))
        del batch_db_clusters, selected_db_clusters

        if current_selected_length > 0:
            selected_mask = complete_mask[:current_selected_length]
            n_selected = sum(selected_mask)
        else:
            n_selected = 0
        batch_mask = complete_mask[current_selected_length:current_selected_length+current_length]
        n_batch = sum(batch_mask)
        if verbose:
            print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")

        # Merge selections
        if current_selected_length > 0:
            selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
            if properties_batch is not None:
                selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
            if return_raw:
                selected_raw_data[:n_selected] = selected_raw_data[:current_selected_length][selected_mask]
        selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
        if return_raw:
            selected_raw_data[n_selected:n_selected+n_batch] = current_raw_data[:current_length][batch_mask]
        if properties_batch is not None:
            selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
        selected_data.flush()
        selected_raw_data.flush() if return_raw else None
        selected_properties.flush() if properties_batch is not None else None
        current_data_filepath = current_data.filename
        current_data._mmap.close()
        os.unlink(current_data_filepath)
        if return_raw:
            current_raw_data_filepath = current_raw_data.filename
            current_raw_data._mmap.close()
            os.unlink(current_raw_data_filepath)
        if properties_batch is not None:
            current_properties_filepath = current_properties.filename
            current_properties._mmap.close()
            os.unlink(current_properties_filepath)

    for key in restore_original.keys():
        setattr(umap_classifier, key, restore_original[key])
        if verbose:
            print(f"Restored original {key} with value {restore_original[key]}.")  
    
    if return_raw:
        return selected_data[:current_selected_length], selected_raw_data[:current_selected_length], selected_properties[:current_selected_length] if properties_batch is not None else None
    else:
        return selected_data[:current_selected_length], None, selected_properties[:current_selected_length] if properties_batch is not None else None