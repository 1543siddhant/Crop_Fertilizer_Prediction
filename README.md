# Crop's Fertilizer Prediction System

Welcome to the Crop's Fertilizer Prediction System! This project aims to help farmers identify the required fertilizer for their crops effortlessly using a machine learning model. By providing key input features such as temperature, humidity, soil type, and crop type, the system can predict the appropriate fertilizer to enhance crop yield and sustainability.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Model](#machine-learning-model)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Crop's Fertilizer Prediction System leverages advanced machine learning algorithms to predict the most suitable fertilizer for different crops based on various environmental and soil parameters. This tool is designed to be user-friendly and provide rapid results to assist farmers in making informed decisions about fertilizer application.

## Features

- **High Accuracy**: Utilizes a RandomForestClassifier model trained on a comprehensive dataset.
- **User-Friendly Interface**: Streamlit-based web application for ease of use.
- **Rapid Results**: Quick analysis and immediate feedback on the recommended fertilizer.
- **Advanced Analysis**: Processes inputs such as temperature, humidity, moisture, soil type, crop type, and nutrient levels.

## Dataset

The dataset used for this project is from the [Fertilizer Prediction dataset](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction/data) available on Kaggle. It contains data on various factors affecting the choice of fertilizer for crops, including:

- Temperature
- Humidity
- Moisture
- Soil Type
- Crop Type
- Nitrogen Levels
- Potassium Levels
- Phosphorus Levels

The target variable is the type of fertilizer recommended for the given conditions.

## Machine Learning Model

The machine learning model was developed with the following key steps:

1. **Data Preprocessing and EDA**:
    - Inspection of the dataset for basic information.
    - Exploratory Data Analysis (EDA) to understand variable distributions and relationships.

2. **Label Encoding**:
    - Encoding of categorical variables (Soil_Type, Crop_Type, Fertilizer) into numerical values.

3. **Data Splitting**:
    - Splitting the dataset into training and test sets using an 80/20 ratio.

4. **Model Training and Evaluation**:
    - Training a RandomForestClassifier on the training data.
    - Hyperparameter tuning using GridSearchCV.
    - Model evaluation using classification metrics such as accuracy, classification report, and confusion matrix.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fertilizer-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd fertilizer-prediction
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Ensure that you have the `classifier.pkl` and `fertilizer.pkl` files in the project directory.

