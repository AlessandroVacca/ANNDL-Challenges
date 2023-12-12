from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np


def evaluate_model(test_labels, predictions):
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(test_labels, predictions)

    # Calculate Mean Squared Error
    mse = mean_squared_error(test_labels, predictions)

    # Calculate Root Mean Squared Error
    rmse = sqrt(mse)

    # Calculate Mean Absolute Percentage Error
    epsilon = 1e-10  # small constant to avoid division by zero
    mape = np.mean(np.abs((test_labels - predictions) / (test_labels + epsilon))) * 100

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")


def split_dataset(dataset, labels, split_percentage=0.8):
    # Calculate the index at which to split the data
    split_index = int(len(dataset) * split_percentage)

    # Split the dataset and labels into training and test sets
    train_data, test_data = dataset[:split_index], dataset[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    print("Train data shape: ", train_data.shape)
    print("Test data shape: ", test_data.shape)
    return train_data, train_labels, test_data, test_labels


def build_sequences_optimized(data, valid_periods, window=200, stride=20, telescope=18):
    assert window % stride == 0

    num_sequences = len(valid_periods)
    dataset = np.zeros((num_sequences, window))
    labels = np.zeros((num_sequences, telescope))

    for i in range(num_sequences):
        start, end = valid_periods[i]
        actual_entry_end = None
        if end - start < window + telescope:
            if end - start < telescope + 1:
                print("Sequence too short, skipping")
                continue
            else:
                entry = data[i, start:end - telescope]
                if entry.shape[0] == 0:
                    print("Sequence too short, skipping it should be impossible")
                    continue
                entry = np.pad(entry, (window - entry.shape[0], 0), 'edge')
                actual_entry_end = end - telescope
        else:
            if end - start - window - telescope == 0:
                actual_start = start
            else:
                actual_start = np.random.randint(start, end - telescope - window)
            entry = data[i, actual_start:actual_start + window]
            actual_entry_end = actual_start + window

        label = data[i, actual_entry_end:actual_entry_end + telescope]
        dataset[i] = entry
        labels[i] = label

    # remove empty rows
    non_empty_indices = ~np.all(dataset == 0, axis=1)
    dataset = dataset[non_empty_indices]
    labels = labels[non_empty_indices]
    print("Dataset shape: ", dataset.shape)
    return dataset, labels
