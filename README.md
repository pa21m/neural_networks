# House Price Prediction using Neural Networks

This project aims to predict house prices using a neural network model built with TensorFlow and Keras. The model is trained on a dataset containing various features of houses, such as the number of bedrooms, bathrooms, square footage, and location. Hyperparameter optimization is performed using Keras Tuner to find the most efficient configuration for the neural network model.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview

This project uses a neural network to predict house prices based on a variety of features. The dataset is cleaned, preprocessed, and feature-engineered to remove outliers, handle missing data, and scale the features. Several models are trained with different configurations of hyperparameters, and Keras Tuner is used to optimize the model architecture and parameters.

## Dependencies

The following libraries are required to run this project:

- pandas
- numpy
- seaborn
- matplotlib
- tensorflow
- keras
- keras-tuner
- scikit-learn

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is the [KC House Data](https://www.kaggle.com/harlfoxem/housesalesprediction), which contains information about 21,613 house sales in King County, USA. The dataset includes columns such as:

- `price`: The sale price of the house.
- `bedrooms`: The number of bedrooms in the house.
- `bathrooms`: The number of bathrooms in the house.
- `sqft_living`: The square footage of the living area.
- `sqft_lot`: The square footage of the lot.
- `zipcode`: The location of the house.

The dataset is processed to handle missing values, remove duplicates, and create new columns to facilitate better predictions.

## Data Preprocessing

The following data preprocessing steps are performed:

1. **Handling Duplicates**: Duplicates are removed based on the house ID to keep the most recent information.
2. **Datetime Conversion**: The `date` column is split into `year`, `month`, and `day` columns.
3. **Feature Creation**: A new column `is_renovated` is added to represent whether a house was renovated or not.
4. **Outlier Removal**: Outliers in the `bedrooms` column are removed.
5. **Feature Scaling**: All numerical features are scaled to the range [0, 1] using MinMaxScaler.
6. **Correlation Analysis**: Highly correlated features (e.g., `sqft_above`) are removed to prevent multicollinearity.

## Model Building

A neural network is built using TensorFlow/Keras with the following structure:

1. **Input Layer**: Flatten the input data into a 1D array.
2. **Hidden Layers**: A fully connected layer with a configurable number of neurons and activation functions (`relu`, `elu`).
3. **Output Layer**: A single output node with a linear activation function to predict the house price.
4. **Compilation**: The model is compiled with an optimizer (`adam` or `nadam`) and a loss function (`mean_squared_error`).

## Hyperparameter Tuning

Keras Tuner is used to perform hyperparameter optimization. The following hyperparameters are tuned:

- **Number of Hidden Units**: The number of neurons in the hidden layers.
- **Activation Function**: The activation function for each layer (`relu`, `elu`).
- **Optimizer**: The optimizer used for training (`adam`, `nadam`).
- **Learning Rate**: The learning rate for the optimizer.
- **Batch Size**: The number of samples per gradient update.
- **Epochs**: The number of times the model will be trained on the dataset.

The optimization is performed using the `RandomSearch` algorithm, which explores different combinations of hyperparameters to find the best model.

## Results

After tuning the hyperparameters, the top-performing models are saved. The models are then evaluated on a test dataset, and the results are visualized:

- **Model Loss**: Training and validation loss are plotted for each model.
- **Predictions vs True Values**: A scatter plot of predicted house prices versus true values is displayed for each of the top models.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**:
   To execute the model training and hyperparameter tuning:
   ```bash
   python house_price_prediction.py
   ```

4. **View TensorBoard**:
   After running the model training, you can visualize the training process with TensorBoard:
   ```bash
   tensorboard --logdir=./logs
   ```

## License

This project is licensed under the MIT License 
