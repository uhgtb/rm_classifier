"""umap_classifier.py - a UMAP classifier with a Y-shaped classification/reconstruction network
================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: All Rights Reserved                    
"""
import os
import sys
root_dir = os.getenv('ROOT_DIR')
sys.path.append(os.path.join(root_dir, 'src'))
print(root_dir)
import joblib
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import ops
from umap import parametric_umap
from src import tracked_parametric_umap
from warnings import warn, filterwarnings
import keras
import tensorflow as tf

class ClassificationValidationCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, validation_loss_fn, metric_name='costum_loss'):
        super().__init__()
        self.validation_data = validation_data
        self.validation_loss_fn = validation_loss_fn
        self.metric_name = metric_name
    
    def on_epoch_end(self, epoch, logs=None):
        
        if logs is None:
            logs = {}    
        X_val, y_val = self.validation_data        
        val_loss = self.validation_loss_fn(X_val, y_val)
        
        logs[self.metric_name] = float(val_loss.numpy())
        print(f" - {self.metric_name}: {logs[self.metric_name]:.4f}")

def create_classification_loss_fn(model_instance):
    classifier_head = getattr(model_instance, 'classifier_head', None)
    encoder = getattr(model_instance, 'encoder', None)
    if classifier_head is None or encoder is None:
        raise ValueError("Classifier head and encoder must be defined in the model instance.")

    def _val_classification_loss_fn(X_val, classes):
        """
        Calculate the validation classification loss.
        """
        classified_mask = classes != -1
        labeled_true_classes = tf.boolean_mask(classes, classified_mask)
        # Get the embeddings for the validation data
        embeddings = encoder.predict(X_val, verbose=0)
        # Get the predicted classes
        y_pred = classifier_head(embeddings)
        labeled_pred_classes = tf.boolean_mask(y_pred, classified_mask, axis=0)

        # Calculate the classification loss
        classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labeled_true_classes, labeled_pred_classes
        )
        # Return the mean classification loss
        return tf.reduce_mean(classification_loss)
    return _val_classification_loss_fn
    
filterwarnings('ignore')

class YShapedUMAPClassifier(tracked_parametric_umap.TrackedPUMAP):
    """
    Y-shaped network that combines UMAP loss with classification loss
    """
    
    def __init__(self, 
                 classifier_head=None,
                 umap_loss_a=1.929, # common parameters for min_dist=0
                 umap_loss_b=0.7915, # common parameters for min_dist=0
                 negative_sample_rate=5,
                 n_classes=100,
                 classification_loss_weight=1.0,
                 keras_fit_kwargs={},
                 **kwargs):
        
        super().__init__(**kwargs)
        self.negative_sample_rate = negative_sample_rate
        self.umap_loss_a = umap_loss_a
        self.umap_loss_b = umap_loss_b
        self.seed_generator = keras.random.SeedGenerator()
        self.classifier_head = classifier_head
        self.n_classes = n_classes
        self.classification_loss_weight = classification_loss_weight
        self.keras_fit_kwargs = keras_fit_kwargs


    def _build_classifier_head(self):
        """Build classification head (one branch of Y)"""
        model = keras.Sequential([
            keras.Input(shape=(self.n_components,)),  # Explicit input layer
            tf.keras.layers.Dense(100),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.Dense(100),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.Dense(100),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.Dense(self.n_classes, activation='softmax', name='classification')
        ])
        return model
    
    def _define_model(self):
        """Define the model in keras"""
        prlw = self.parametric_reconstruction_loss_weight
        self.parametric_model = yUMAPModel(
            self._a,
            self._b,
            negative_sample_rate=self.negative_sample_rate,
            encoder=self.encoder,
            decoder=self.decoder,
            classification_loss_weight=self.classification_loss_weight,
            parametric_reconstruction_loss_fn=self.parametric_reconstruction_loss_fcn,
            parametric_reconstruction=self.parametric_reconstruction,
            parametric_reconstruction_loss_weight=prlw,
            global_correlation_loss_weight=self.global_correlation_loss_weight,
            autoencoder_loss=self.autoencoder_loss,
            classifier_head=self.classifier_head,
        )
    def save(self, filepath):
        # Temporarily break the circular reference
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
            os.makedirs(filepath, exist_ok=True)
            super().save(filepath)
            if self._history:
                joblib.dump(self._history, os.path.join(filepath, 'history.pkl'))
        finally:
            # Restore the original callbacks
            if original_callbacks is not None:
                for cb in original_callbacks:
                    if hasattr(cb, '_original_loss_fn'):
                        cb.validation_loss_fn = cb._original_loss_fn
                        delattr(cb, '_original_loss_fn')
                self.parametric_model._callbacks = original_callbacks
        os.makedirs(filepath, exist_ok=True)
        super().save(filepath)
        if self._history:
            joblib.dump(self._history, os.path.join(filepath, 'history.pkl'))

    def transform(self, X, **kwargs):
        embs = super().transform(X, **kwargs)
        probs = self.classifier_head(embs)
        classes = tf.argmax(probs, axis=1)
        return embs, probs, classes
    
    def fit_transform(self, X, classes=None, validation_data=None, val_train_split=0.2,**kwargs):
        """
        validation_data: tuple of (X_val, classes_val)
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        classes : array, shape (n_samples,)
            Class labels for the samples in X. Used for classification tasks.
        val_train_split : float, optional
            Proportion of training data to use for validation if no validation_data is provided. Default is 0.2.
        """
        if classes is None:
            raise ValueError("Classes must be provided for classification tasks.")
        self.n_classes = len(np.unique(classes))            
        # Build the classification network
        self.classifier_head = self._build_classifier_head()
        self.val_train_split = val_train_split         
        if isinstance(validation_data, type(None)):
            print(f"Warning: No validation data provided. Taking {self.val_train_split*100}% of training data for validation.") 
            X_train, X_val, classes_train, classes_val = train_test_split(X,classes, test_size=self.val_train_split)
            self.train_classes = tf.convert_to_tensor(classes_train, dtype=tf.int16)
            self.val_classes = tf.convert_to_tensor(classes_val, dtype=tf.int16)
            self.X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            super().fit_transform(X_train, validation_data=tf.convert_to_tensor(X_val, dtype=tf.float32), **kwargs)
            return self.encoder.predict(X)
        else:
            self.X_val = tf.convert_to_tensor(validation_data[0], dtype=tf.float32)
            self.train_classes = tf.convert_to_tensor(classes, dtype=tf.int16)
            self.val_classes = tf.convert_to_tensor(validation_data[1], dtype=tf.int16)
            return super().fit_transform(X, validation_data=tf.convert_to_tensor(validation_data[0], dtype=tf.float32),**kwargs)


    def _construct_edge_dataset(self,
        X,
        graph_,
        n_epochs,
        batch_size,
        parametric_reconstruction,
        global_correlation_loss_weight,
        landmark_positions=None,
        classes=None
    ):
        """
        Construct a tf.data.Dataset of edges, sampled by edge weight.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        graph_ : scipy.sparse.csr.csr_matrix
            Generated UMAP graph
        n_epochs : int
            # of epochs to train each edge
        batch_size : int
            batch size
        parametric_reconstruction : bool
            Whether the decoder is parametric or non-parametric
        landmark_positions : array, shape (n_samples, n_components), optional
            The desired position in low-dimensional space of each sample in X.
            Points that are not landmarks should have nan coordinates.
        classes : array, shape (n_samples,), optional
            Class labels for the samples in X. Used for classification tasks.
        """

        def gather_index(tensor, index):
            return tensor[index]

        # if X is > 512Mb in size, we need to use a different, slower method for
        #    batching data.
        gather_indices_in_python = True if X.nbytes * 1e-9 > 0.5 else False
        if landmark_positions is not None:
            gather_landmark_indices_in_python = (
                True if landmark_positions.nbytes * 1e-9 > 0.5 else False
            )

        def gather_X(edge_to, edge_from):
            # gather data from indexes (edges) in either numpy of tf, depending on array size
            if gather_indices_in_python:
                edge_to_batch = tf.py_function(gather_index, [X, edge_to], [tf.float32])[0]
                edge_from_batch = tf.py_function(
                    gather_index, [X, edge_from], [tf.float32]
                )[0]
                classes_to_batch = tf.py_function(gather_index, [classes, edge_to], [tf.int16])[0]
            else:
                edge_to_batch = tf.gather(X, edge_to)
                edge_from_batch = tf.gather(X, edge_from)
                classes_to_batch = tf.gather(classes, edge_to)
            return edge_to, edge_from, edge_to_batch, edge_from_batch, classes_to_batch

        def get_outputs(edge_to, edge_from, edge_to_batch, edge_from_batch, classes_to_batch):
            outputs = {"umap": ops.repeat(0, batch_size)}
            if global_correlation_loss_weight > 0:
                outputs["global_correlation"] = edge_to_batch
            if parametric_reconstruction:
                # add reconstruction to iterator output
                # edge_out = ops.concatenate([edge_to_batch, edge_from_batch], axis=0)
                outputs["reconstruction"] = edge_to_batch
            if landmark_positions is not None:
                if gather_landmark_indices_in_python:
                    outputs["landmark_to"] = tf.py_function(
                        gather_index, [landmark_positions, edge_to], [tf.float32]
                    )[0]
                else:
                    # Make sure we explicitly cast landmark_positions to float32,
                    # as it's user-provided and needs to play nice with loss functions.
                    outputs["landmark_to"] = tf.gather(landmark_positions, edge_to)
            return (edge_to_batch, edge_from_batch, classes_to_batch), outputs

        # get data from graph
        _, epochs_per_sample, head, tail, weight, n_vertices = parametric_umap.get_graph_elements(
            graph_, n_epochs
        )

        # number of elements per batch for embedding
        if batch_size is None:
            # batch size can be larger if its just over embeddings
            batch_size = int(np.min([n_vertices, 1000]))

        edges_to_exp, edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )

        # shuffle edges
        shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
        edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
        edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)

        # create edge iterator
        edge_dataset = tf.data.Dataset.from_tensor_slices((edges_to_exp, edges_from_exp))
        edge_dataset = edge_dataset.repeat()
        edge_dataset = edge_dataset.shuffle(10000)
        edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
        edge_dataset = edge_dataset.map(
            gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        edge_dataset = edge_dataset.map(
            get_outputs, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        edge_dataset = edge_dataset.prefetch(10)

        return edge_dataset, batch_size, len(edges_to_exp), head, tail, weight

    
    def _fit_embed_data(self, X, n_epochs, init, random_state, landmark_positions=None):

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
            warn(
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
        ) = self._construct_edge_dataset(
            X,
            self.graph_,
            self.n_epochs,
            self.batch_size,
            self.parametric_reconstruction,
            self.global_correlation_loss_weight,
            classes=self.train_classes,
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

        # Validation dataset 
        # for reconstruction
        if (
            self.parametric_reconstruction
            and self.reconstruction_validation is not None
        ):

            # reshape data for network
            if len(self.dims) > 1:
                self.reconstruction_validation = np.reshape(
                    self.reconstruction_validation,
                    [len(self.reconstruction_validation)] + list(self.dims),
                )

            validation_data = (
                (
                    self.reconstruction_validation,
                    ops.zeros_like(self.reconstruction_validation),
                ),
                {"reconstruction": self.reconstruction_validation},
            )
        else:
            validation_data = None
        X_val = self.X_val
        y_val = self.val_classes    
        validation_data = (X_val, y_val)
        _val_classification_loss_fn = create_classification_loss_fn(self)
        if validation_data is not None:
            custom_val_callback = ClassificationValidationCallback(
                validation_data=validation_data,
                validation_loss_fn=_val_classification_loss_fn,
                metric_name='val_classification_loss'
            )
            self.keras_fit_kwargs['callbacks'].insert(0,custom_val_callback)
            self.keras_fit_kwargs['verbose'] = 1
        ###test###
        y_train = self.train_classes    
        train_class_data = (X, y_train)
        _train_classification_loss_fn = create_classification_loss_fn(self)
        if validation_data is not None:
            custom_train_callback = ClassificationValidationCallback(
                validation_data=train_class_data,
                validation_loss_fn=_train_classification_loss_fn,
                metric_name='train_classification_loss'
            )
            self.keras_fit_kwargs['callbacks'].insert(0,custom_train_callback)
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
        self._history = history.history

        # get the final embedding
        embedding = self.encoder.predict(X, verbose=self.verbose)

        return embedding, {}

class yUMAPModel(parametric_umap.UMAPModel):
    def __init__(self,         
                 umap_loss_a,
                 umap_loss_b,
                 negative_sample_rate,
                 encoder,
                 decoder,
                 classifier_head=None,
                 classification_loss_weight=1.0,
                 **kwargs):
        super().__init__(umap_loss_a, umap_loss_b, negative_sample_rate, encoder, decoder, **kwargs)
        self.classifier_head = classifier_head
        self.classification_loss_weight = classification_loss_weight

    
    def _classification_loss(self, y_pred):

        classified_mask = y_pred["true_class"] != -1
        labeled_true_classes = tf.boolean_mask(y_pred["true_class"], classified_mask)
        labeled_pred_classes = tf.boolean_mask(y_pred["pred_class"], classified_mask, axis=0)

        classification_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labeled_true_classes, labeled_pred_classes
            )
        )
        return self.classification_loss_weight * classification_loss
    @tf.function
    def call(self, inputs):
        to_x, from_x, classes = inputs
        embedding_to = self.encoder(to_x)
        embedding_from = self.encoder(from_x)
        y_pred = {
            "embedding_to": embedding_to,
            "embedding_from": embedding_from,
        }
        classification = self.classifier_head(embedding_to)
        y_pred["pred_class"] = classification
        y_pred["true_class"] = classes
        return y_pred
    
    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs
    ):
        losses = []
        # Regularization losses.
        for loss in self.losses:
            losses.append(ops.cast(loss, dtype=keras.backend.floatx()))

        # umap loss
        losses.append(self._umap_loss(y_pred))

        # classification loss
        losses.append(self._classification_loss(y_pred))

        # global correlation loss
        if self.global_correlation_loss_weight > 0:
            losses.append(self._global_correlation_loss(y, y_pred))

        # parametric reconstruction loss
        if self.parametric_reconstruction:
            losses.append(self._parametric_reconstruction_loss(y, y_pred))
        return ops.sum(losses)
    
