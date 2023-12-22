import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/model'))

    def predict(self, X, categories):
        
        # Note: this is just an example.
        # Here the model.predict is called
        out = self.model.predict(X)[:,:9]  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2

        return out