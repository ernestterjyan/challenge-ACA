# Physionet-challenge-ACA
# Machine Learning Model for Predicting In-Hospital Death

## Overview
This repository contains a machine learning project focused on predicting in-hospital death based on various patient parameters. The project includes data preprocessing, model training, and prediction steps, structured across different Python files.

## Project Structure
- `preprocessor.py`: Contains the `Preprocessor` class for data preprocessing, including NaN value imputation and PCA.
- `model.py`: Contains the `Model` class that encapsulates the ensemble model used for predictions.
- `run_pipeline.py`: Includes the `Pipeline` class with a `run` method to handle training and testing modes.
- `model_selection.py`: (Optional) Contains functionality for selecting the best models and hyperparameters.
- `README.md`: This file, providing an overview and instructions for the project.

## Best Models and Hyperparameters
The following models were identified as the best through grid search, with their respective hyperparameters and scores:

- **Logistic Regression (LR)**
  - Parameters: `{'LR__C': 0.1, 'pca__n_components': 30}`
  - Score: `0.82589`

- **Random Forest (RF)**
  - Parameters: `{'RF__max_depth': 10, 'RF__n_estimators': 100, 'pca__n_components': 30}`
  - Score: `0.81461`

- **XGBoost (XGB)**
  - Parameters: `{'XGB__gamma': 0.5, 'XGB__learning_rate': 0.1, 'XGB__max_depth': 3, 'XGB__subsample': 0.8, 'pca__n_components': 30}`
  - Score: `0.82955`

- **Naive Bayes (NB)**
  - Parameters: `{'pca__n_components': 15}`
  - Score: `0.76134`

- **K-Nearest Neighbors (KNN)**
  - Parameters: `{'KNN__n_neighbors': 7, 'pca__n_components': 15}`
  - Score: `0.73862`

- **Support Vector Machine (SVM)**
  - Parameters: `{'SVM__C': 0.1, 'SVM__kernel': 'linear', 'pca__n_components': 30}`
  - Score: `0.81082`

- **Artificial Neural Network (ANN)**
  - Parameters: `{'ANN__activation': 'relu', 'ANN__alpha': 0.0001, 'ANN__hidden_layer_sizes': (50,), 'pca__n_components': 15}`
  - Score: `0.80767`

## Ensemble Model Results
The ensemble model combining LR, RF, and XGB achieved the following results on the test set:
- Accuracy: `0.90125`
- ROC-AUC: `0.94322`

## Usage
### Training the Model
Run `run_pipeline.py` with the `--data_path` argument pointing to your training data:

```bash
python run_pipeline.py --data_path "path/to/training_data.csv" --save_model
