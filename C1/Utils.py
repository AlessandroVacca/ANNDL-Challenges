import numpy as np
import matplotlib.pyplot as plt


def plot_misclassified_images(X, y_true, model, n_plot, train_history):
    # Plot the re-trained and the transfer learning MobileNetV2 training histories
    plt.figure(figsize=(15, 5))
    plt.plot(train_history['loss'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(train_history['val_loss'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15, 5))
    plt.plot(train_history['accuracy'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(train_history['val_accuracy'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()
    # Evaluate the model on the test set
    test_accuracy = model.evaluate(X, y_true, verbose=0)[-1]
    print('Test set accuracy %.4f' % test_accuracy)

    # Find the misclassified indices
    y_pred = model.predict(X)

    y_true = np.argmax(y_true, axis=1)
    y_pred_arg_max = np.argmax(y_pred, axis=1)
    misclassified_indices = np.where(y_true != y_pred_arg_max)[0]
    # Number of misclassified images to plot
    n = min(n_plot, len(misclassified_indices))

    # Map labels to human-readable names (adjust this mapping as needed)
    label_mapping = {0: "healthy", 1: "unhealthy"}

    # Plot the first 'n' misclassified images
    plt.figure(figsize=(30, 14))
    for i in range(n):
        plt.subplot(2, n // 2, i + 1)
        index = misclassified_indices[i]
        plt.imshow(X[index])
        true_label = label_mapping[y_true[index]]
        predicted_label = label_mapping[y_pred_arg_max[index]]
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}\n Probabilities: {y_pred[index]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    correct_classified_indices = np.where(y_true == y_pred_arg_max)[0]
    # Number of misclassified images to plot
    n = min(n_plot, len(correct_classified_indices))

    # Map labels to human-readable names (adjust this mapping as needed)
    label_mapping = {0: "healthy", 1: "unhealthy"}

    # Plot the first 'n' misclassified images
    plt.figure(figsize=(30, 14))
    for i in range(n):
        plt.subplot(2, n // 2, i + 1)
        index = correct_classified_indices[i]
        plt.imshow(X[index])
        true_label = label_mapping[y_true[index]]
        predicted_label = label_mapping[y_pred_arg_max[index]]
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}\n Probabilities: {y_pred[index]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
