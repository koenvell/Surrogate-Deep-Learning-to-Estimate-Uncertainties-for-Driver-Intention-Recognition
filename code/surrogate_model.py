import tensorflow as tf
import numpy as np
tfk = tf.keras
import logging
logging.root.setLevel(logging.INFO)


class SurrogateModel:
    def __init__(self,
                 n_sequence: int = 7,
                 n_features: int = 16,
                 n_hidden: int = 60,
                 p_rate: int = .3,
                 n_labels: int = 5,
                 uc_output: str = "class_uc",
                 batch_size : int = 128,
                 epochs : int = 1000, 
                 verbose: int = 1,
                 uc_loss_weight: float = 0.5):
        self.n_sequence = n_sequence
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.p_rate = p_rate
        self.n_labels = n_labels
        self.uc_output_dict = {"total_uc": self.uc_output_total, 
                               "class_uc": self.uc_output_class,
                              "class_sp": self.uc_output_class_softplus}
        self.uc_output = uc_output
        
        self.uc_loss = {f"uncertainty_estimation_class_{str(i)}":"mean_squared_error" for i in range(0,n_labels)}
        self.losses = {"intention_classifier": "categorical_crossentropy"} | self.uc_loss
        
        self.uc_lossWeights = {f"uncertainty_estimation_class_{str(i)}":uc_loss_weight for i in range(0,n_labels)}
        self.lossWeights = {"intention_classifier": 1.0} | self.uc_lossWeights
      
        self.metrics = {"intention_classifier":[tfk.metrics.CategoricalCrossentropy(),
                       tfk.metrics.Precision(), tfk.metrics.Recall(), tfk.metrics.CategoricalAccuracy()],}
        self.callbacks = [tfk.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
    
        
    def build_surrogate_model(self):
        """
        Args:
            None
        
        Returns:
            None
        """
        inputs = tfk.layers.Input(shape=(self.n_sequence, self.n_features), name="input")
        x = tfk.layers.LSTM(units=self.n_hidden, 
                            dropout=self.p_rate,
                            return_sequences=True,
                            name="LSTM_1")(inputs)
        x = tfk.layers.LSTM(units=self.n_hidden, 
                            dropout=self.p_rate,
                            return_sequences=False,
                            name="LSTM_2")(x)
        outputs_clf = tfk.layers.Dense(units=self.n_labels, activation="softmax", name="intention_classifier")(x)
        outputs_uep = self.uc_output_layer(x) #[tfk.layers.Dense(units=1,   name=f"uncertainty_estimation_class_{str(i)}")(x) for i in range(self.n_labels)]
        output_layers = [outputs_clf] +  outputs_uep

        self.model = tfk.Model(inputs=inputs, outputs=output_layers)
        logging.info(f"  Model succesfully constructed  ".center(50,"#"))
#         return self.model
    
    def compile_model(self):
        """
        Args:
            None
        
        Returns:
            None
        """
        self.model.compile(loss=self.losses, 
                           metrics=self.metrics,
                           optimizer='Adam',
                           loss_weights=self.lossWeights)
        
        logging.info(f"  Model compiled  ".center(50,"#"))
        
    def uc_output_total(self, x):
        """ Add a single output head for the uncertainty estimation to the surrogate network.
        
        Args:
            None
        
        Returns:
            None
        """
        return tfk.layers.Dense(units=1,  name=f"uncertainty_estimation_total")
    
    def uc_output_class(self, x):
        """ Add a ouput head per class to estimate the uncertainty.
        Args:
            x (tfk.layer): last feature layer
        
        Returns:
            None
        """
        return [tfk.layers.Dense(units=1, 
                                 name=f"uncertainty_estimation_class_{str(i)}")(x) for i in range(self.n_labels)]
    def uc_output_class_softplus(self, x):
        """ Add a ouput head per class to estimate the uncertainty.
        Args:
            x (tfk.layer): last feature layer
        
        Returns:
            None
        """
        return [tfk.layers.Dense(units=1, activation="softplus",
                                 name=f"uncertainty_estimation_class_{str(i)}")(x) for i in range(self.n_labels)]
    
    def uc_output_layer(self,x):
        """ Adds the desired uncertainty estimation output head based on uc_output.  
        Args:
            None
        
        Returns:
            None
        """
        return self.uc_output_dict[self.uc_output](x)


    def train_surrogate_model(self, X_train, y_train, y_uc):
        """Wrapper to run the training of a surrogate uncertainty estimation model
        
        Args:
            x_train (np.array): shape(i,j,k), where i=num 
            y_train (np.array): shape(m,n)
        Returns: 
        """
        self.build_surrogate_model()
        self.compile_model()
        
        y_uc_class = {f"uncertainty_estimation_class_{str(i)}":np.asanyarray(y_uc)[:,i:i+1] for i in range(0, self.n_labels)}
        y_total = {"intention_classifier": y_train} | y_uc_class 
        
        logging.info(f"  Start model training  ".center(50,"#"))
        self.hist = self.model.fit(x=X_train, 
                       y=y_total,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=self.callbacks,
                       verbose=self.verbose)
        
    def predict(self, X_test, ):
        self.predictions = self.model.predict_on_batch(X_test)
       
        logging.info(f"  Predicted on testset  ".center(50, "#"))
    
    def save(self, filename: str = 'tmp.npy'):
        # create array with all estimates
        predictions = self.predictions
        sur_uc = np.squeeze(np.dstack((predictions[1],predictions[2],predictions[3],predictions[4],predictions[5],)))
        np.save(filename, np.array([predictions[0], sur_uc]))
