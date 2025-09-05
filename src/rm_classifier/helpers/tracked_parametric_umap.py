"""
tracked_parametric_umap - Parametric UMAP with training tracking and model saving/loading.

Based on/Extends: UMAP (Uniform Manifold Approximation and Projection)
Original Author: Leland McInnes
Original Source: https://github.com/lmcinnes/umap
License: 3-clause BSD License

Modifications by: Johann Luca Kastner
Date: 15/09/2025
License: MIT

Description: This module extends the parametric UMAP implementation to include
additional features such as tracking training history, saving/loading models,
and computing validation loss during training.
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import ops
import umap.parametric_umap as parametric_umap
import warnings

class AutoencoderLossCallback(keras.callbacks.Callback):
    """Callback to compute and log autoencoder loss on any dataset (train/val) during training."""
    def __init__(self, data, loss_fn, metric_name='autoencoder_loss'):
        """Initialize the callback.

        Args:
            validation_data (list, np.ndarray): Validation data.
            loss_fn (Callable): Validation loss function.
            metric_name (str, optional): name of the callbacks metric. Defaults to 'val_autoencoder_loss'.
        """
        super().__init__()
        self.data = data
        self.loss_fn = loss_fn
        self.metric_name = metric_name

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        X = self.data
        loss = self.loss_fn(X)

        logs[self.metric_name] = float(loss.numpy())
        print(f" - {self.metric_name}: {logs[self.metric_name]:.4f}")

def _create_autoencoder_loss_fn(model_instance):
    """ Create a autoencoder loss function for a specific TrackedPUMAP model.

    Args:
        model_instance (TrackedPUMAP): the TrackedPUMAP model instance.

    Returns:
        Callable: autoencoder loss function.
    """
    loss_fn = getattr(model_instance, 'parametric_reconstruction_loss_fcn', None)
    if loss_fn is None:
        warnings.warn("No reconstruction loss function defined in the model instance. The default MeanSquaredError will be used.")
        loss_fn = keras.losses.MeanSquaredError()
    encoder = getattr(model_instance, 'encoder', None)
    decoder = getattr(model_instance, 'decoder', None)

    def _val_autoencoder_loss_fn(X_val):
        """
        Calculate the validation reconstruction loss.
        """
        embs= encoder(X_val, training=False)
        reconstructed = decoder(embs, training=False)
        autoencoder_loss = loss_fn(X_val, reconstructed)
        if isinstance(autoencoder_loss, tuple):
            autoencoder_loss = autoencoder_loss[0]
        if isinstance(autoencoder_loss, tf.Tensor):
            autoencoder_loss = tf.reduce_mean(autoencoder_loss)
        elif isinstance(autoencoder_loss, np.ndarray):
            autoencoder_loss = np.mean(autoencoder_loss)
        elif isinstance(autoencoder_loss, float):
            autoencoder_loss = tf.constant(autoencoder_loss, dtype=tf.float32)
        else:
            raise TypeError(f"Unexpected type for autoencoder_loss: {type(autoencoder_loss)}")
        return autoencoder_loss
    return _val_autoencoder_loss_fn



class TrackedPUMAP(parametric_umap.ParametricUMAP):
    """
    Subclass of ParametricUMAP that tracks training history and allows for saving/loading.
    """
    
    def __init__(self, 
                 n_training_epochs=1,
                 repulsion_strength=1.0,
                 loss_report_frequency=1000,
                 **kwargs):
        """Initialize the TrackedPUMAP model.

        Args:
            n_training_epochs (int, optional): number of training epochs. Defaults to 1.
            repulsion_strength (float, optional): repulsion strength. Defaults to 1.0.
            loss_report_frequency (int, optional): loss report frequency. Defaults to 1000.
        """
        
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator()
        self.loss_report_frequency = loss_report_frequency
        self.repulsion_strength = repulsion_strength
        self.n_training_epochs = n_training_epochs


    def save(self, dir_path):
        """Save the model to a specified dir_path.

        Args:
            dir_path (string): path to the directory where the model should be saved.
        """
        original_callbacks = None
        
        if hasattr(self, 'parametric_model') and self.parametric_model is not None:
            model = self.parametric_model
            if hasattr(model, '_callbacks'):
                original_callbacks = model._callbacks
                # Create new callbacks without the circular reference
                safe_callbacks = []
                for cb in model._callbacks:
                    if hasattr(cb, 'validation_loss_fn'):
                        # Replace the method reference with None temporarily
                        cb._original_loss_fn = cb.validation_loss_fn
                        cb.validation_loss_fn = None
                    safe_callbacks.append(cb)
                model._callbacks = safe_callbacks
        
        try:
            # Now save
            os.makedirs(dir_path, exist_ok=True)
            super().save(dir_path)
            if self._history:
                joblib.dump(self._history, os.path.join(dir_path, 'history.pkl'))
        finally:
            # Restore the original callbacks
            if original_callbacks is not None:
                for cb in original_callbacks:
                    if hasattr(cb, '_original_loss_fn'):
                        cb.validation_loss_fn = cb._original_loss_fn
                        delattr(cb, '_original_loss_fn')
                self.parametric_model._callbacks = original_callbacks
        os.makedirs(dir_path, exist_ok=True)
        super().save(dir_path)
        if self._history:
            joblib.dump(self._history, os.path.join(dir_path, 'history.pkl'))

    def _define_model(self):
        """Define the model in keras"""
        prlw = self.parametric_reconstruction_loss_weight
        if self.parametric_reconstruction_loss_fcn is None and self.parametric_reconstruction:
            warnings.warn("No reconstruction loss function specified. Using MeanSquaredError as default.")
            self.parametric_reconstruction_loss_fcn = keras.losses.MeanSquaredError()
        self.parametric_model = TrackedUMAPModel(
            self._a,
            self._b,
            negative_sample_rate=self.negative_sample_rate,
            encoder=self.encoder,
            decoder=self.decoder,
            parametric_reconstruction_loss_fn=self.parametric_reconstruction_loss_fcn,
            parametric_reconstruction=self.parametric_reconstruction,
            parametric_reconstruction_loss_weight=prlw,
            global_correlation_loss_weight=self.global_correlation_loss_weight,
            autoencoder_loss=self.autoencoder_loss,
            repulsion_strength=self.repulsion_strength,
        )

    def fit_transform(self, X, validation_data=None, val_train_split=0.2, **kwargs):
        """Fit the model to the data and transform it.

        Args:
            X (list, np.ndarray): input data.
            validation_data (list, np.array, optional): data used for validation. Defaults to None.
            val_train_split (float, optional): percentage of validation data from X if validation_data is None. Defaults to 0.2.

        Returns:
            np.ndarray: array of transformed data.
        """
        self.val_train_split = val_train_split         
        if isinstance(validation_data, type(None)):
            print(f"Warning: No validation data provided. Taking {self.val_train_split*100}% of training data for validation.")
            X_train, X_val = train_test_split(X, test_size=self.val_train_split, shuffle=True)
            print(f"Using {len(X_val)} samples for validation.")
            self.X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            super().fit_transform(X_train, **kwargs)
            return self.encoder.predict(X)
        else:
            self.X_val = tf.convert_to_tensor(validation_data, dtype=tf.float32)
            return super().fit_transform(X, **kwargs)

    
    def _fit_embed_data(self, X, n_epochs, init, random_state, **kwargs):
        if self.metric == "precomputed":
            X = self._X

        # get dimensionality of dataset
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]
        else:
            # reshape data for network
            if len(self.dims) > 1:
                X = np.reshape(X, [len(X)] + list(self.dims))

        if self.parametric_reconstruction and (np.max(X) > 1.0 or np.min(X) < 0.0):
            warnings.warn(
                "Data should be scaled to the range 0-1 for cross-entropy reconstruction loss."
            )

        # get dataset of edges
        (
            edge_dataset,
            self.batch_size,
            n_edges,
            head,
            tail,
            self.edge_weight,
        ) = parametric_umap.construct_edge_dataset(
            X,
            self.graph_,
            self.n_epochs,
            self.batch_size,
            self.parametric_reconstruction,
            self.global_correlation_loss_weight,
        )
        self.head = ops.array(ops.expand_dims(head.astype(np.int64), 0))
        self.tail = ops.array(ops.expand_dims(tail.astype(np.int64), 0))

        init_embedding = None

        # create encoder and decoder model
        n_data = len(X)
        self.encoder, self.decoder = parametric_umap.prepare_networks(
            self.encoder,
            self.decoder,
            self.n_components,
            self.dims,
            n_data,
            self.parametric_reconstruction,
            init_embedding,
        )

        # create the model
        self._define_model()

        # report every loss_report_frequency subdivision of an epochs
        steps_per_epoch = int(
            n_edges / self.batch_size / self.loss_report_frequency
        )
        #Tracking history
        if self.parametric_reconstruction:
            _val_autoencoder_loss_fn = _create_autoencoder_loss_fn(self)

            if self.X_val is not None:
                
                autoencoder_val_callback = AutoencoderLossCallback(
                    data=self.X_val,
                    loss_fn=_val_autoencoder_loss_fn,
                    metric_name='val_autoencoder_loss'
                )

                # Create training loss callback
                autoencoder_train_callback = AutoencoderLossCallback(
                    data=tf.convert_to_tensor(X, dtype=tf.float32),
                    loss_fn=_val_autoencoder_loss_fn,  # same function, same model
                    metric_name='train_autoencoder_loss'
                )

                # Clean old callbacks of same type (optional)
                self.keras_fit_kwargs['callbacks'] = [
                    cb for cb in self.keras_fit_kwargs.get('callbacks', [])
                    if not isinstance(cb, AutoencoderLossCallback)
                ]

                # Insert both callbacks
                self.keras_fit_kwargs['callbacks'].insert(0, autoencoder_val_callback)
                self.keras_fit_kwargs['callbacks'].insert(0, autoencoder_train_callback)

                self.keras_fit_kwargs['verbose'] = 1

        # create embedding
        history = self.parametric_model.fit(
            edge_dataset,
            epochs=self.n_training_epochs,
            batch_size=self.batch_size,
            steps_per_epoch=steps_per_epoch,
            **self.keras_fit_kwargs
        )
        # save loss history dictionary
        if self.parametric_reconstruction:
            history.history["train_umap_loss"]=list(np.array(history.history["loss"])-self.parametric_reconstruction_loss_weight*np.array(history.history["train_autoencoder_loss"]))
        else:
            history.history["train_umap_loss"]=history.history["loss"]
        self._history = history.history

        # get the final embedding
        embedding = self.encoder.predict(X, verbose=self.verbose)

        return embedding, {}
    
class TrackedUMAPModel(parametric_umap.UMAPModel):
    """
    A UMAP model that tracks the training history and allows for saving/loading.
    """
    
    def __init__(self,
                 umap_loss_a,
                 umap_loss_b,
                 negative_sample_rate,
                 encoder,
                 decoder,
                 repulsion_strength=1.0,
                 **kwargs):
        """Initialize the TrackedUMAPModel.

        Args:
            umap_loss_a (float): UMAP parameter a for the definition of similarity in the latent space
            umap_loss_b (float): UMAP parameter b for the definition of similarity in the latent space
            negative_sample_rate (int): the number of negative samples to use per positive sample. If not specified, defaults to 5.
            encoder (keras.src.models.sequential.Sequential): encoder model
            decoder (keras.src.models.sequential.Sequential): decoder model
            repulsion_strength (float, optional): repulsion strength for the calculation of the umap loss. Defaults to 1.0.
        """
        super().__init__(umap_loss_a,
                 umap_loss_b,
                 negative_sample_rate,
                 encoder,
                 decoder,**kwargs)
        self.repulsion_strength = repulsion_strength
    def _umap_loss(self, y_pred):
        # split out to/from
        repulsion_strength=self.repulsion_strength
        embedding_to = y_pred["embedding_to"]
        embedding_from = y_pred["embedding_from"]

        # get negative samples
        embedding_neg_to = ops.repeat(embedding_to, self.negative_sample_rate, axis=0)
        repeat_neg = ops.repeat(embedding_from, self.negative_sample_rate, axis=0)

        repeat_neg_batch_dim = ops.shape(repeat_neg)[0]
        shuffled_indices = keras.random.shuffle(
            ops.arange(repeat_neg_batch_dim), seed=self.seed_generator)

        if keras.config.backend() == "tensorflow":
            embedding_neg_from = tf.gather(
                repeat_neg, shuffled_indices
            )
        else:
            embedding_neg_from = repeat_neg[shuffled_indices]

        #  distances between samples (and negative samples)
        distance_embedding = ops.concatenate(
            [
                ops.norm(embedding_to - embedding_from, axis=1),
                ops.norm(embedding_neg_to - embedding_neg_from, axis=1),
            ],
            axis=0,
        )

        # convert distances to probabilities
        log_probabilities_distance = parametric_umap.convert_distance_to_log_probability(
            distance_embedding, self.umap_loss_a, self.umap_loss_b
        )

        # set true probabilities based on negative sampling
        batch_size = ops.shape(embedding_to)[0]
        probabilities_graph = ops.concatenate(
            [
                ops.ones((batch_size,)),
                ops.zeros((batch_size * self.negative_sample_rate,)),
            ],
            axis=0
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = parametric_umap.compute_cross_entropy(
            probabilities_graph,
            log_probabilities_distance,
            repulsion_strength=repulsion_strength,
        )
         # return mean cross entropy loss
        return ops.mean(ce_loss)
    
    def _parametric_reconstruction_loss(self, y, y_pred):
        loss = self.parametric_reconstruction_loss_fn(
            y["reconstruction"], y_pred["reconstruction"]
        )
        return loss * self.parametric_reconstruction_loss_weight