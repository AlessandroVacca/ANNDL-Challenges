import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/resnet3blocks.h5'))

    def predict(self, X, categories):
        # Note: this is just an example.
        # Here the model.predict is called
        X_reshaped = np.expand_dims(X, axis=-1)

        out = self.model.predict(X_reshaped)  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
        out = out[:, :9]

        return out

    
    
    
