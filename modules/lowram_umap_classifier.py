"""
low_ram_umap_classifier.py - a UMAP classifier especially usable in low RAM use-cases
================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: All Rights Reserved
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import os
import umap_classifier
from src import statistical_models, visualization, high_entropy_subset
import umap.umap_ as umap
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import joblib
import copy
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

class LowRAM_UMAPClassifier(umap_classifier.UMAPClassifier):
    
    def __init__(self, 
                 **kwargs
                 ):
        super().__init__(**kwargs)

    def load_umap_data_from_dir(self, dir_path, data_loader = None, data_loader_kwargs = {}, 
                                max_umap_events = 1e5, batch_size = None, n_selected_per_batch = None, summary_batch_size = None,
                                local_embedding = True,
                                umap_kwargs = {}, clustering_kwargs = {}, clustering_mode = "dbscan",
                                file_extension = ".pkl", verbose=True):
        

        if local_embedding:
            if "n_neighbors" not in umap_kwargs.keys():
                umap_kwargs["n_neighbors"] = 50
            if clustering_mode == "dbbscan":
                if "eps" not in clustering_kwargs.keys():
                    clustering_kwargs["eps"] = 0.4

        memmap_flag = True
        def default_data_loader(file_path):
            df = joblib.load(file_path)
            return np.stack(df[self.input_data_type].to_numpy()), None
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
                n_batches = int(np.ceil(data.shape[0]/batch_size))
                for i in range(n_batches):
                    if verbose:
                        print(f"Processing batch {i+1}/{n_batches}...")
                    i_min = i*batch_size
                    i_max = min((i+1)*batch_size, data.shape[0])
                    data_batch = data[i_min:i_max]
                    properties_batch = None if properties is None else properties[i_min:i_max]
                    prepared_data = self.prepare_data(data_batch, verbose=verbose)
                    del data_batch
                    embeddings=self.embed(prepared_data, **umap_kwargs, verbose=verbose)
                    if clustering_mode == "dbscan":
                        db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    elif clustering_mode == "hdbscan":
                        db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    else:
                        raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                    batch_selected_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, min(n_selected_per_batch, i_max-i_min))
                    if verbose:
                        print(f"Selected {sum(batch_selected_mask)} events, current total {current_length}/{summary_batch_size}.")

                    if memmap_flag:
                        os.makedirs(os.path.join(self.dir, "umap_loaded_data"), exist_ok=True)
                        self.umap_data_dir = os.path.join(self.dir, "umap_loaded_data")
                        current_data = np.memmap(os.path.join(self.umap_data_dir, "current_data.dat"), dtype=np.int32, mode='w+', shape=(summary_batch_size,prepared_data.shape[1]))
                        current_properties = np.zeros((summary_batch_size,properties.shape[1]))
                        selected_data = np.memmap(os.path.join(self.umap_data_dir, "selected_data.dat"), dtype=np.int32, mode='w+', shape=(max_umap_events,prepared_data.shape[1]))
                        selected_properties = np.zeros((max_umap_events,properties.shape[1]))
                        memmap_flag = False


                    current_data[current_length:current_length + sum(batch_selected_mask)] = prepared_data[batch_selected_mask]
                    if properties is not None:
                        current_properties[current_length:current_length + sum(batch_selected_mask)] = properties_batch[batch_selected_mask]
                    current_length += sum(batch_selected_mask)
                    del prepared_data
                    del embeddings
                    del db_clusters
                    del batch_selected_mask

                    # Find clusters in current selection
                    if current_length > summary_batch_size-n_selected_per_batch:
                        if verbose:
                            print(f"Reached {current_length} events, performing final selection...")
                        if current_selected_length > 0:
                            embeddings=self.embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
                            if clustering_mode == "dbscan":
                                selected_db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                            elif clustering_mode == "hdbscan":
                                selected_db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                            else:
                                raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                            del embeddings
                            _, counts = np.unique(selected_db_clusters, return_counts=True)
                            selected_entropy = entropy(counts)
                            n_selected_noise = sum(selected_db_clusters == -1)
                        else:
                            selected_entropy = 0
                            n_selected_noise = 0
                            selected_db_clusters = np.array([])

                        # Find clusters in current batch
                        embeddings=self.embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
                        if clustering_mode == "dbscan":
                            db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        elif clustering_mode == "hdbscan":
                            db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        else:
                            raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                        del embeddings
                        _, counts = np.unique(db_clusters, return_counts=True)
                        batch_entropy = max(entropy(counts), 1e-5)
                        n_batch_noise = sum(db_clusters == -1)

                        # Determine how many to select from batch vs previous selection
                        n_batch = max([n_batch_noise, int(batch_entropy/(batch_entropy + selected_entropy)*max_umap_events)])
                        n_batch = min([n_batch, current_length, max_umap_events - n_selected_noise])
                        n_selected = min(max_umap_events - n_batch, current_selected_length)
                        if verbose:
                            print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")
                        batch_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, n_batch)
                        if current_selected_length > 0:
                            selected_mask = high_entropy_subset.high_entropy_subset_mask(selected_db_clusters, n_selected)
                        del db_clusters, selected_db_clusters

                        # Merge selections
                        if current_selected_length > 0:
                            selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
                            if properties is not None:
                                selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
                        selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
                        if properties is not None:
                            selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
                        selected_data.flush()
                        current_selected_length = n_selected + n_batch
                        current_length = 0
                        current_data[:] = 0                        
        # Finalize selection
        if verbose:
            print(f"Reached the end of data loading, performing final selection...")
        if current_length > 0:
             # Find clusters in current selection
            if current_selected_length > 0:
                embeddings=self.embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
                if clustering_mode == "dbscan":
                    selected_db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                elif clustering_mode == "hdbscan":
                    selected_db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                else:
                    raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                del embeddings
                _, counts = np.unique(selected_db_clusters, return_counts=True)
                selected_entropy = entropy(counts)
                n_selected_noise = sum(selected_db_clusters == -1)
            else:
                selected_entropy = 0
                n_selected_noise = 0
                selected_db_clusters = np.array([])

            # Find clusters in current batch
            embeddings=self.embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
            if clustering_mode == "dbscan":
                db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
            elif clustering_mode == "hdbscan":
                db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
            else:
                raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
            del embeddings
            _, counts = np.unique(db_clusters, return_counts=True)
            batch_entropy = max(entropy(counts), 1e-5)
            n_batch_noise = sum(db_clusters == -1)

            # Determine how many to select from batch vs previous selection
            n_batch = max([n_batch_noise, int(batch_entropy/(batch_entropy + selected_entropy)*max_umap_events)])
            n_batch = min([n_batch, current_length, max_umap_events - n_selected_noise])
            n_selected = min(max_umap_events - n_batch, current_selected_length)
            if verbose:
                print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")
            batch_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, n_batch)
            if current_selected_length > 0:
                selected_mask = high_entropy_subset.high_entropy_subset_mask(selected_db_clusters, n_selected)
            del db_clusters, selected_db_clusters

            # Merge selections
            if current_selected_length > 0:
                selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
                if properties is not None:
                    selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
            selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
            if properties is not None:
                selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
            selected_data.flush()
            current_data_filepath = current_data.filename
            current_data._mmap.close()
            os.unlink(current_data_filepath)                      

        return selected_data[:current_selected_length], selected_properties[:current_selected_length] if properties is not None else None    
  
                
        
    
    
    def load_umap_data_batchwise_from_dir(self, dir_path, data_batch_loader_kwargs = {}, data_batch_loader = None,
                                max_umap_events = 1e5, batch_size = None, n_selected_per_batch = None, summary_batch_size = None,
                                local_embedding = True,
                                umap_kwargs = {}, clustering_kwargs = {}, clustering_mode = "dbscan",
                                file_extension = ".pkl", verbose=True):
        

        if local_embedding:
            if "n_neighbors" not in umap_kwargs.keys():
                umap_kwargs["n_neighbors"] = 50
            if clustering_mode == "dbbscan":
                if "eps" not in clustering_kwargs.keys():
                    clustering_kwargs["eps"] = 0.4

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
                return np.stack(df[self.input_data_type].to_numpy())[i_min:i_max], None, True
            else:
                return np.stack(df[self.input_data_type].to_numpy())[i_min:i_max], None, False
        
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
                    
                    prepared_data = self.prepare_data(data_batch, verbose=verbose)
                    del data_batch
                    embeddings=self.embed(prepared_data, **umap_kwargs, verbose=verbose)
                    if clustering_mode == "dbscan":
                        db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    elif clustering_mode == "hdbscan":
                        db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                    else:
                        raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                    batch_selected_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, min(n_selected_per_batch, len(db_clusters)))
                    if verbose:
                        print(f"Selected {sum(batch_selected_mask)} events, current total {current_length}/{summary_batch_size}.")

                    if memmap_flag:
                        os.makedirs(os.path.join(self.dir, "umap_loaded_data"), exist_ok=True)
                        self.umap_data_dir = os.path.join(self.dir, "umap_loaded_data")
                        current_data = np.memmap(os.path.join(self.umap_data_dir, "current_data.dat"), dtype=np.int32, mode='w+', shape=(summary_batch_size,prepared_data.shape[1]))
                        current_properties = np.zeros((summary_batch_size,properties_batch.shape[1]))
                        selected_data = np.memmap(os.path.join(self.umap_data_dir, "selected_data.dat"), dtype=np.int32, mode='w+', shape=(max_umap_events,prepared_data.shape[1]))
                        selected_properties = np.zeros((max_umap_events,properties_batch.shape[1]))
                        memmap_flag = False


                    current_data[current_length:current_length + sum(batch_selected_mask)] = prepared_data[batch_selected_mask]
                    if properties_batch is not None:
                        current_properties[current_length:current_length + sum(batch_selected_mask)] = properties_batch[batch_selected_mask]
                    current_length += sum(batch_selected_mask)
                    del prepared_data
                    del embeddings
                    del db_clusters
                    del batch_selected_mask

                    # Find clusters in current selection
                    if current_length > summary_batch_size-n_selected_per_batch:
                        if verbose:
                            print(f"Reached {current_length} events, performing final selection...")
                        if current_selected_length > 0:
                            embeddings=self.embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
                            if clustering_mode == "dbscan":
                                selected_db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                            elif clustering_mode == "hdbscan":
                                selected_db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                            else:
                                raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                            del embeddings
                            _, counts = np.unique(selected_db_clusters, return_counts=True)
                            selected_entropy = entropy(counts)
                            n_selected_noise = sum(selected_db_clusters == -1)
                        else:
                            selected_entropy = 0
                            n_selected_noise = 0
                            selected_db_clusters = np.array([])

                        # Find clusters in current batch
                        embeddings=self.embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
                        if clustering_mode == "dbscan":
                            db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        elif clustering_mode == "hdbscan":
                            db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                        else:
                            raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                        del embeddings
                        _, counts = np.unique(db_clusters, return_counts=True)
                        batch_entropy = max(entropy(counts), 1e-5)
                        n_batch_noise = sum(db_clusters == -1)

                        # Determine how many to select from batch vs previous selection
                        n_batch = max([n_batch_noise, int(batch_entropy/(batch_entropy + selected_entropy)*max_umap_events)])
                        n_batch = min([n_batch, current_length, max_umap_events - n_selected_noise])
                        n_selected = min(max_umap_events - n_batch, current_selected_length)
                        if verbose:
                            print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")
                        batch_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, n_batch)
                        if current_selected_length > 0:
                            selected_mask = high_entropy_subset.high_entropy_subset_mask(selected_db_clusters, n_selected)
                        del db_clusters, selected_db_clusters

                        # Merge selections
                        if current_selected_length > 0:
                            selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
                            if properties_batch is not None:
                                selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
                        selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
                        if properties_batch is not None:
                            selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
                        selected_data.flush()
                        current_selected_length = n_selected + n_batch
                        current_length = 0
                        current_data[:] = 0                        
        # Finalize selection
        if verbose:
            print(f"Reached the end of data loading, performing final selection...")
        if current_length > 0:
             # Find clusters in current selection
            if current_selected_length > 0:
                embeddings=self.embed(selected_data[:current_selected_length], **umap_kwargs, verbose=verbose)
                if clustering_mode == "dbscan":
                    selected_db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
                elif clustering_mode == "hdbscan":
                    selected_db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
                else:
                    raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
                del embeddings
                _, counts = np.unique(selected_db_clusters, return_counts=True)
                selected_entropy = entropy(counts)
                n_selected_noise = sum(selected_db_clusters == -1)
            else:
                selected_entropy = 0
                n_selected_noise = 0
                selected_db_clusters = np.array([])

            # Find clusters in current batch
            embeddings=self.embed(current_data[:current_length], **umap_kwargs, verbose=verbose)
            if clustering_mode == "dbscan":
                db_clusters=self.db_classify(embeddings, **clustering_kwargs, verbose=verbose)
            elif clustering_mode == "hdbscan":
                db_clusters=self.hdb_classify(embeddings, **clustering_kwargs, verbose=verbose)
            else:
                raise ValueError(f"Cluster mode {clustering_mode} not recognized. Use 'dbscan' or 'hdbscan'.")
            del embeddings
            _, counts = np.unique(db_clusters, return_counts=True)
            batch_entropy = max(entropy(counts), 1e-5)
            n_batch_noise = sum(db_clusters == -1)

            # Determine how many to select from batch vs previous selection
            n_batch = max([n_batch_noise, int(batch_entropy/(batch_entropy + selected_entropy)*max_umap_events)])
            n_batch = min([n_batch, current_length, max_umap_events - n_selected_noise])
            n_selected = min(max_umap_events - n_batch, current_selected_length)
            if verbose:
                print(f"Selecting {n_batch} from batch and {n_selected} from previous selection (total {n_batch+n_selected}/{max_umap_events}).")
            batch_mask = high_entropy_subset.high_entropy_subset_mask(db_clusters, n_batch)
            if current_selected_length > 0:
                selected_mask = high_entropy_subset.high_entropy_subset_mask(selected_db_clusters, n_selected)
            del db_clusters, selected_db_clusters

            # Merge selections
            if current_selected_length > 0:
                selected_data[:n_selected] = selected_data[:current_selected_length][selected_mask]
                if properties_batch is not None:
                    selected_properties[:n_selected] = selected_properties[:current_selected_length][selected_mask]
            selected_data[n_selected:n_selected+n_batch] = current_data[:current_length][batch_mask]
            if properties_batch is not None:
                selected_properties[n_selected:n_selected+n_batch] = current_properties[:current_length][batch_mask]
            selected_data.flush()
            current_data_filepath = current_data.filename
            current_data._mmap.close()
            os.unlink(current_data_filepath)  
        
        return selected_data[:current_selected_length], selected_properties[:current_selected_length] if properties_batch is not None else None