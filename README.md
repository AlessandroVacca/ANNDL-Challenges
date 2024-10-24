# Challenges for the Artificial Neural Networks course A.Y. 2023-24
Repository for the challenges of the Artificial Neural Network and Deep Learning course, held at  Politecnico di Milano for the A.Y. 2023/2024. The code for both projects is implemented in Python and relies mainly on the Keras and TensorFlow libraries.
## Description of Challenge 1
This project focuses on an image recognition problem, a sub-field of computer vision that involves building Machine Learning models capable of interpreting and understanding the semantic information contained in an image. More in detail, the task is to develop Convolutional Neural Networks (CNNs) to classify the health condition of plants based on a picture. Thus, this can be considered a binary classification problem, where the goal is to distinguish between healthy and unhealthy plants.
### Overview of our solution

This project processes the dataset and builds an ensemble of models for optimal performance:

1. **Dataset Processing**:
   - **PCA Dimensionality Reduction**: Features from a pretrained **MobileNetV2** were reduced using PCA.
   - **Outlier & Duplicate Removal**: Outliers were removed with Mahalanobis distance, and duplicates were eliminated, reducing the dataset from 5200 to 4850 images.
   - **Data Augmentation**: Using `imgaug`, augmentations like random affine transformations, cropping, noise addition, and color adjustments were applied to handle class imbalance and improve generalization.

2. **Model Architecture**:
   - **Ensemble Models**: An ensemble of **Xception**, **EfficientNetV2S**, **ResNet50V2**, **ConvNeXtBase**, and **DenseNet121** was used, leveraging ImageNet pre-trained weights.
   - **Fine-Tuning**: Selective fine-tuning of deeper layers was done to adapt models to the dataset.
   - **Optimizer**: Models were trained with the **ADAM optimizer**, with hyperparameters fine-tuned via a systematic search.
For more information, please read the final report.
## Description of Challenge 2
This projects consists in a timeseries forecasting problem, which involves analyzing timeseries data using statistics and modelling to predict the future values of a variable of interest based on its historical observations. The dataset consisted of multiple univariate timeseries from six different domains: demography, finance, industry, macroeconomy, microeconomy, and others. The objective was to build a model that could generalize well across different timeseries, predicting the next values in the sequence accurately. Specifically, the model needed to process a timeseries of length 200 and predict the next 9-18 values.
### Overview of our solution

The best-performing model in this time series forecasting task was a **weighted average ensemble** of several architectures, including both standard and autoregressive versions. Here are the key aspects:

1. **Model Components**:
   - **LSTM**: A simple, high-performing LSTM model with two layers (LSTM and dense output).
   - **RESNET**: A Residual Network architecture with 1D convolutions and skip connections.
   - **LSTM Encoder-Decoder**: Combines bidirectional LSTM layers with skip connections.
   - **Attention Model**: An LSTM model enhanced with an attention mechanism to prioritize key parts of the input sequence.
   - **Convolutional Model**: Utilizes convolutional layers for feature extraction, followed by LSTM layers for sequence understanding.

2. **Weighted Averaging**:
   - The ensemble used a **weighted average** approach, where the predictions of each model were combined based on optimized weights.
   - Weights were determined using a **softmin function** based on the local validation set performance, allowing the ensemble to focus on models that performed better on the dataset.

3. **Performance**:
   - This ensemble achieved the best result with a **Mean Squared Error (MSE) of 0.0090**, outperforming both individual models and simpler ensemble methods like basic averaging.
   - The ensemble effectively leveraged the strengths of different models, balancing the diversity of architectures and minimizing prediction errors.

For more information, please read the final report.
## Authors
 - *[Paolo Piccirilli](https://github.com/PaoloPiccirilli)*
 - *[Marco Riva](https://github.com/MarcoRiva6)*
 - *[Federico Sarrocco](https://github.com/FedeAi)*
 - *[Alessandro Vacca](https://github.com/AlessandroVacca)*
