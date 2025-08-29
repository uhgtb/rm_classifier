#import custom modules
import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import umap_classifier
import numpy as np
import pandas as pd
import src.tracked_parametric_umap as tracked_parametric_umap
import joblib

class ParametricUMAPClassifier(umap_classifier.UMAPClassifier):
    def __init__(self, repulsion_strength = 1.0,
                 n_training_epochs=100,
                 loss_report_frequeny=1000, 
                 **kwargs):
        """
        Initialize the Parametric UMAP Classifier with the given parameters.
        """
        super().__init__(**kwargs)
        self.parametric_model = None
        self.repulsion_strength = repulsion_strength
        self.n_training_epochs = n_training_epochs
        self.loss_report_frequeny = loss_report_frequeny

    def embed(self, data, 
              keep_model=True, 
              save_model = None, 
              verbose=True,
              umap_kwargs={},
              **kwargs):
        for key, value in kwargs.items():
            if verbose:
                print(f"Overriding {key} with value {value} from kwargs.")
            self.__setattr__(key, value)
        if verbose:
            print(f"Embedding data with parameters: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, n_components={self.n_components}, metric={self.metric}")
        
        umap_model = tracked_parametric_umap.TrackedPUMAP(n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist, 
            n_components= self.n_components, 
            metric = self.metric, 
            verbose=verbose,
            repulsion_strength=self.repulsion_strength,
            n_training_epochs=self.n_training_epochs,
            loss_report_frequency=self.loss_report_frequeny,
            #**umap_kwargs
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