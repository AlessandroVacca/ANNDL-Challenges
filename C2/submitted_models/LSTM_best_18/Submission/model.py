import os
import tensorflow as tf


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/model.h5'))

    def predict(self, X, categories):
        # Note: this is just an example.
        # Here the model.predict is called
        out = self.model.predict(X)  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
        out = out[:, :9]

        return out
