import numpy as np


# Define augmentation functions
def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def add_scaling(data, scaling_range=(0.8, 1.2)):
    scaling = np.random.uniform(*scaling_range)
    return data * scaling


def add_constant(data, constant_range=(-0.1, 0.1)):
    constant = np.random.uniform(*constant_range)
    return data + constant


def add_seasonal_variation(data):
    seasonal_amplitude = np.random.uniform(0.1, 0.3)
    seasonal_period = np.random.randint(20, len(data) // 4)
    time_points = np.arange(0, len(data))
    seasonal_component = np.sin(2 * np.pi * time_points / seasonal_period) * seasonal_amplitude
    return data + seasonal_component


# Augmentation function using augmentation types
# def augment_data(train_data, train_labels, num_augmentations=3):
#     augmentation_functions = [add_noise, add_scaling, add_constant, add_seasonal_variation]
#     num_augmentation_types = len(augmentation_functions)
#     total_augmentations = num_augmentations * num_augmentation_types
#
#     augmented_train_data = np.empty((total_augmentations * len(train_data), *train_data.shape[1:]))
#     augmented_train_labels = np.empty((total_augmentations * len(train_labels), *train_labels.shape[1:]))
#
#     for n in range(num_augmentations):
#         print("Augmentation round:", n)
#         for i, data_point in enumerate(train_data):
#             start_idx = n * num_augmentation_types * len(train_data) + i * num_augmentation_types
#
#             for idx, augmentation_func in enumerate(augmentation_functions):
#                 augmented_train_data[start_idx + idx] = augmentation_func(data_point)
#                 augmented_train_labels[start_idx + idx] = train_labels[i]
#
#     return augmented_train_data, augmented_train_labels

# Augmentation function using augmentation types
def augment_data(train_data, train_labels, num_augmentation_types=3, num_augmentations=3):
    augmentation_functions = [add_noise, add_scaling, add_constant, add_seasonal_variation]
    total_augmentations = num_augmentations * num_augmentation_types

    augmented_train_data = np.empty((total_augmentations * len(train_data), *train_data.shape[1:]))
    augmented_train_labels = np.empty((total_augmentations * len(train_labels), *train_labels.shape[1:]))

    for n in range(num_augmentations):
        print("Augmentation round:", n)
        for i, data_point in enumerate(train_data):
            # Concatenate data_point and train_labels
            combined_data = np.concatenate((data_point, train_labels[i]), axis=-1)
            selected_transforms = np.random.choice(augmentation_functions, num_augmentation_types, replace=False)

            for idx, augmentation_func in enumerate(selected_transforms):
                start_idx = n * num_augmentation_types * len(train_data) + i * num_augmentation_types
                augmented_combined = augmentation_func(combined_data)

                # Separate augmented_combined into data and labels
                augmented_train_data[start_idx + idx] = augmented_combined[:train_data.shape[1]]
                augmented_train_labels[start_idx + idx] = augmented_combined[train_data.shape[1]:]

    return augmented_train_data, augmented_train_labels


if __name__ == "__main__":
    # test augmentation functions
    import matplotlib.pyplot as plt
    data = np.sin(np.arange(0, 100)/10)
    plt.plot(data)
    plt.plot(add_noise(data))
    plt.plot(add_scaling(data))
    plt.plot(add_constant(data))
    plt.plot(add_seasonal_variation(data))
    plt.legend(["Original", "Noise", "Scaling", "Constant", "Seasonal"])
    plt.show()

    # test augmentation function, data are sine waves
    n_datapoints = 48000
    data = np.sin(np.arange(n_datapoints * 218).reshape((n_datapoints, 218))/10)
    train_data = data[:, :200]
    train_labels = data[:, 200:]

    augmented_train_data, augmented_train_labels = augment_data(train_data, train_labels)
    print(augmented_train_data.shape)
    print(augmented_train_labels.shape)
    plt.figure()
    plt.plot(train_data[0])
    # make the plots size bigger
    plt.plot(range(len(augmented_train_data[0])), augmented_train_data[0], linewidth=2)
    plt.plot(range(len(augmented_train_data[0]), len(augmented_train_data[0]) + len(augmented_train_labels[0])), augmented_train_labels[0], linewidth=2)
    plt.legend(["Original", "Augmented"])
    plt.show()


