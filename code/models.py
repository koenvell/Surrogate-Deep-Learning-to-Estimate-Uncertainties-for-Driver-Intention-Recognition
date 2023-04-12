"""Implementation of monte carlo dropout, deep ensembles, 
   stochastich weight averaging - gaussian (SWA-G), multi-SWAG,
   bayes by backprp (bbb/variational inference approximation)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["KVELLENG"]
__version__ = "0.0.1"
__maintainer__ = "Koen Vellenga"
__status__ = "Dev"
__file__ = "./prob_dl.py"



from  tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import edward2 as ed  # named after George Edward Pelham Box
import numpy as np
from purestochastic.model import swag 

from glob import glob
from utilities import B4C_dataset


class TrainParams:
    def __init__(self,
                 n_train: int = 1799,
                 n_hidden: int = 60,
                 n_labels: int = 5,
                 n_sequence: int = 7,
                 n_features: int = 16,
                 p_rate:float = 0.3,
                 batch_size: int = 128,
                 lr: int = 1e-3,
                 epochs: int = 1000,
                 start_avg: int = 100,
                 update_freq:int = 5, 
                 n_predict: int = 100,
                 optimizer: object = keras.optimizers.Adam,
                 metrics: list = [keras.metrics.CategoricalCrossentropy(),
                                  keras.metrics.Precision(),
                                  keras.metrics.Recall(),
                                  keras.metrics.CategoricalAccuracy(),],
                callbacks: list = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                                   patience=50,
                                                                   restore_best_weights=True)]) -> None:
        self.n_train = n_train
        self.n_hidden = n_hidden
        self.n_labels =  n_labels
        self.n_sequence = n_sequence
        self.n_features = n_features
        self.p_rate = p_rate
        self.batch_size = batch_size
        self.loss = keras.losses.CategoricalCrossentropy()
        self.lr = lr
        self.epochs = epochs
        self.start_avg = start_avg
        self.update_freq = update_freq
        self.n_predict = n_predict
        self.optimizer = optimizer(learning_rate=lr)
        self.metrics = metrics
        self.callbacks = callbacks

class MC_LSTM(keras.layers.LSTM):
    """MC LSTM dropout layer

    Source:
        https://datascience.stackexchange.com/questions/48030/how-to-apply-mc-dropout-to-an-lstm-network-kera
        https://proceedings.neurips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf

    Args:
        tf ([type]): [description]

    Returns:
        tf.keras.layers.LSTM: LSTM layer that uses dropout during test time
    """

    def call(self, inputs):
        """[summary]
        """
        return super().call(inputs, training=True)    

class VariationalInference(TrainParams):
    """Wrapper to create tf VI layers.
    """
    def __init__(self) -> None:
        TrainParams.__init__(self)
        self.kl_div = (lambda q, p, _: tfp.distributions.kl_divergence(q, p)
                       / tf.cast(self.n_train, dtype=tf.float32))

    def VI_LSTM(self, 
                return_seq = False):
        """VI LSTM layer implementation.

        Args:
            return_seq (Bool): returns 2d Tensor if false, 3d if true. Defaults to False.
        """
        # first initiate the RNN layer and then put the probabilistic cell inside
        return keras.layers.RNN(
                ed.layers.LSTMCellFlipout(
                    units=self.n_hidden,
                    dropout=self.p_rate,
                    kernel_regularizer=ed.regularizers.NormalKLDivergence(scale_factor=1./self.n_train),
                    recurrent_regularizer=ed.regularizers.NormalKLDivergence(scale_factor=1./self.n_train),),
                    return_sequences=return_seq)
    
    def VI_output(self):
        """Probabilistic output layer regularlized based on the kl_div 
        """
        return tfp.layers.DenseFlipout(self.n_labels, 
                                       activation="softmax",
                                       kernel_divergence_fn=self.kl_div)

    def VI_model(self):
        inputs = keras.layers.Input(shape=(self.n_sequence,
                                    self.n_features))
        x = self.VI_LSTM(return_seq=True)(inputs)
        x = self.VI_LSTM(return_seq=False)(x)
        outputs = self.VI_output()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)     
        return model

    def VI_LL_model(self):
        inputs = keras.layers.Input(shape=(self.n_sequence, self.n_features))
        x = keras.layers.LSTM(units=self.n_hidden, 
                            dropout=self.p_rate,
                            return_sequences=True)(inputs)
        x = keras.layers.LSTM(units=self.n_hidden, 
                            dropout=self.p_rate,
                            return_sequences=False)(x)
        outputs = self.VI_output()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)     
        return model


class ModelBuilder(TrainParams):
    def __init__(self) -> None:
        TrainParams.__init__(self)
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model :
        """ Build LSTM model
        Returns:
            keras.Model
        """
        inputs = keras.layers.Input(shape=(self.n_sequence, self.n_features))
        x = keras.layers.LSTM(units=self.n_hidden, 
                            dropout=self.p_rate,
                            return_sequences=True)(inputs)
        x = keras.layers.LSTM(units=self.n_hidden, 
                            dropout=self.p_rate,
                            return_sequences=False)(x)
        outputs = keras.layers.Dense(units=self.n_labels, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)     
        return model

    def baseline(self):
        """Compile the base model.
        """
        model = self._build_model()   
        return self._model_compile(model)

    def mc_dropout(self):
        """
        """
        inputs = keras.layers.Input(shape=(self.n_sequence, self.n_features))
        x = MC_LSTM(units=self.n_hidden, 
                    dropout=self.p_rate,
                    return_sequences=True)(inputs)
        x = MC_LSTM(units=self.n_hidden, 
                    dropout=self.p_rate,
                    return_sequences=False)(x)
        outputs = keras.layers.Dense(units=self.n_labels, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)     
        return self._model_compile(model)

    def vi(self):
        """Bayes approximation: Variational inference model
        """
        return self._model_compile(VariationalInference().VI_model())

    def vi_ll(self):
        """Bayes approximation: Variational inference model
        """
        return self._model_compile(VariationalInference().VI_LL_model())

    def swa_g(self):
        """ Stochastic weight averaging wrapper
        """
        model = self._build_model()
        return self._model_compile(swag.toSWAG(model))
    
    def multi_swa_g(self, nb_models: int = 10):
        """ Ensemble of stochastic weight averaging models
        
        Args: 
            nb_models (int): number of models 
        """
        model = self._build_model()
        swag_model = self._model_compile(swag.toSWAG(model))
        multi_swag = DeepEnsembleClassifier(swag_model, n_models=nb_models, swag=True)
        return multi_swag
        

    def deep_ensemble(self, nb_models: int = 100) -> object:
        """Deep ensemble wrapper for the baseline model

        Args: 
            nb_models (int): number of models
        """
        model = self._build_model()
        return DeepEnsembleClassifier(model, n_models=nb_models) 
        
    def _model_compile(self, model) -> keras.Model:
        """Compile keras functional model

        Args:
            inputs (KerasTensor): Input layer of the model.
            outpus (KerasTensor): Output layer of the model.

        Returns:
            keras.Model: compiles the Keras model based on the TrainingParams.
        """
        # Compile the model.
        model.compile(loss=self.loss,
                      metrics=self.metrics,
                      optimizer=self.optimizer)
        return model
