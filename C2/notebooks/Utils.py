from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from C2.notebooks.data_augmentation import augment_data


def evaluate_model(predictions, test_labels):
    # Calculate Mean Absolute Error
    predictions = predictions.flatten()
    test_labels = test_labels.flatten()
    mae = mean_absolute_error(test_labels, predictions)

    # Calculate Mean Squared Error
    mse = mean_squared_error(test_labels, predictions)

    # Calculate Root Mean Squared Error
    rmse = sqrt(mse)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")


def split_dataset(dataset, labels, categories = None, split_percentage=0.8, augment_train_data=False, num_augmentations=3):
    # Split the dataset sklearn.model_selection.train_test_split
    from sklearn.model_selection import train_test_split
    train_categories = None
    test_categories = None
    if categories is not None:
        train_data, test_data, train_labels, test_labels, train_categories, test_categories = train_test_split(dataset, labels, categories, shuffle=True, train_size=split_percentage)

    else:
        train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, shuffle=True, train_size=split_percentage)

    if augment_train_data:
        print("Augmenting training data")
        train_data, train_labels, train_categories = augment_data(train_data, train_labels, categories=train_categories, num_augmentations=num_augmentations)

    print("Train data shape: ", train_data.shape)
    print("Test data shape: ", test_data.shape)
    return train_data, train_labels, test_data, test_labels, train_categories, test_categories


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


def plot_predictions(test_data, predictions, test_labels, series_index):
    # Select the series to plot
    n_test_data_to_plot = 4 * predictions.shape[1]
    series_test_data = test_data[series_index]
    series_predictions = predictions[series_index]
    series_test_labels = test_labels[series_index]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_test_data_to_plot), series_test_data[-n_test_data_to_plot:], label='Data')
    plt.plot(range(n_test_data_to_plot, n_test_data_to_plot + predictions.shape[1]), series_test_labels, label='Actual')
    plt.plot(range(n_test_data_to_plot, n_test_data_to_plot + predictions.shape[1]), series_predictions,
             label='Predicted')

    # Add title and labels
    plt.title(f'Time Series {series_index} - Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
