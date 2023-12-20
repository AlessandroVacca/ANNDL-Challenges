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
        predictions_all = np.zeros((len(X), 18, len(models) * 4))

        dataset = X
        for i, model_name in enumerate(models):
            model = tf.keras.models.load_model(self.path + model_name)
            if "cat" in model_name.lower():
                preds = model.predict([dataset, categories], batch_size=512 * 2)
                predictions_all[:, :, i] = preds
            elif "step9" in model_name.lower():
                preds1 = model.predict(dataset, batch_size=512 * 2)
                expanded_dataset = np.concatenate((dataset, preds1), axis=1)
                expanded_dataset = expanded_dataset[:, -200:]
                preds2 = model.predict(expanded_dataset, batch_size=512 * 2)
                preds = np.concatenate((preds1, preds2), axis=1)

                predictions_all[:, :, i] = preds
            else:
                preds = model.predict(dataset, batch_size=512 * 2)
                predictions_all[:, :, i] = preds

            predictions_all[:, :, i] = preds

        # autoregressive
        window = 3
        prediction_len = 18
        assert prediction_len % window == 0

        for i, model_name in enumerate(models):
            j = len(models) + i
            model = tf.keras.models.load_model(self.path + model_name)
            expanded_dataset = dataset
            if "cat" in model_name.lower():
                for _ in range(prediction_len // window):
                    pre = model.predict([expanded_dataset, categories], batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-prediction_len + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds
            elif "step9" in model_name.lower():
                for _ in range(prediction_len // window):
                    pre = model.predict(expanded_dataset, batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-9 + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds
            else:
                for _ in range(prediction_len // window):
                    pre = model.predict(expanded_dataset, batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-prediction_len + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds

        window = 6

        for i, model_name in enumerate(models):
            j = len(models) * 2 + i
            model = tf.keras.models.load_model(self.path + model_name)
            expanded_dataset = dataset
            if "cat" in model_name.lower():
                for _ in range(prediction_len // window):
                    pre = model.predict([expanded_dataset, categories], batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-prediction_len + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds
            elif "step9" in model_name.lower():
                for _ in range(prediction_len // window):
                    pre = model.predict(expanded_dataset, batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-9 + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds
            else:
                for _ in range(prediction_len // window):
                    pre = model.predict(expanded_dataset, batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-prediction_len + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds

        window = 9
        for i, model_name in enumerate(models):
            j = len(models) * 3 + i
            model = tf.keras.models.load_model(self.path + model_name)
            expanded_dataset = dataset
            if "cat" in model_name.lower():
                for _ in range(prediction_len // window):
                    pre = model.predict([expanded_dataset, categories], batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-prediction_len + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds
            elif "step9" in model_name.lower():
                for _ in range(prediction_len // window):
                    pre = model.predict(expanded_dataset, batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds
            else:
                for _ in range(prediction_len // window):
                    pre = model.predict(expanded_dataset, batch_size=512 * 2)
                    expanded_dataset = np.concatenate((expanded_dataset, pre), axis=1)
                    expanded_dataset = expanded_dataset[:, :-prediction_len + window]
                    expanded_dataset = expanded_dataset[:, -200:]
                preds = expanded_dataset[:, -prediction_len:]
                predictions_all[:, :, j] = preds

        out = np.mean(predictions_all, axis=2)

        return out