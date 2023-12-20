import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


class model:
    def __init__(self, path):
        self.path = os.path.join(path, 'SubmissionModel/')

    def predict(self, X, categories):
        
        # Note: this is just an example.
        # Here the model.predict is called
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(categories.reshape(-1, 1))
        categories = enc.transform(categories.reshape(-1, 1)).toarray()

        # ENSEMBLE
        models = [file_name for file_name in os.listdir(self.path)]
        if len(models) == 0:
            raise Exception("No models found in " + self.path)

        # ENSEMBLE MODEL average
        predictions = np.zeros((len(X), 18))
        predictions_all = np.zeros((len(X), 18, len(models)))
        for i, model_name in enumerate(models):
            model = tf.keras.models.load_model(self.path + model_name)
            if "cat" in model_name.lower():
                preds = model.predict([X, categories])
                predictions += preds / len(models)
                predictions_all[:, :, i] = preds
            elif "step9" in model_name.lower():
                preds1 = model.predict(X)
                expanded_dataset = np.concatenate((X, preds1), axis=1)
                expanded_dataset = expanded_dataset[:, -200:]
                preds2 = model.predict(expanded_dataset)
                preds = np.concatenate((preds1, preds2), axis=1)

                predictions += preds / len(models)
                predictions_all[:, :, i] = preds
            else:
                preds = model.predict(X) / len(models)
                predictions += preds / len(models)
                predictions_all[:, :, i] = preds


        out = predictions

        return out