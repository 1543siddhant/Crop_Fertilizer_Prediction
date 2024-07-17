import streamlit as st
import numpy as np
import pickle
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Load the models for fertilizer prediction
classifier_model = pickle.load(open('classifier.pkl', 'rb'))
label_encoder = pickle.load(open('fertilizer.pkl', 'rb'))

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Fertilizer Prediction"])

# Mapping for soil and crop types
soil_mapping = {
    "Black": 0,
    "Clayey": 1,
    "Loamy": 2,
    "Red": 3,
    "Sandy": 4
}

crop_mapping = {
    "Barley": 0,
    "Cotton": 1,
    "Ground Nuts": 2,
    "Maize": 3,
    "Millets": 4,
    "Oil Seeds": 5,
    "Paddy": 6,
    "Pulses": 7,
    "Sugarcane": 8,
    "Tobacco": 9,
    "Wheat": 10
}

# Main Page
if app_mode == "Home":
    st.header("FERTILIZER PREDICTION SYSTEM")
    image_path = "home.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    # 🌿 Welcome to the Crop's Fertilizer Prediction System! 🔍

Identify the required Crop's Fertilizer effortlessly with our advanced prediction system. Upload the values, and let our technology do the rest.

---

## 🚀 Get Started
1. **Upload Your data**: Navigate to the **Crop's Fertilizer Prediction Page**.
2. **Instant Analysis**: Our system processes your image with state-of-the-art algorithms.
3. **Receive Results**: Get immediate feedback and recommendations.

---

## 🌟 Why Choose Our System?
- **High Accuracy**: Leveraging cutting-edge machine learning for precise Fertilizer identification.
- **User-Friendly Interface**: Designed for ease of use.
- **Rapid Results**: Quick analysis for timely decision-making.

---

## 📋 Steps to Use
1. **Visit the Crop's Fertilizer Prediction Page**
2. **Upload the required values regarding your crop**
3. **Wait for the System to Analyze**
4. **View Results and Recommendations**

---

/* ## 📖 Contributors. 
                
### [Siddhant Patil](https://1543siddhant.github.io/Portfolio/)
### [Rahul Raut](https://codebyte156.github.io/home/)

Discover more about our mission, team, and the technology behind the system on the **About** page.

Join us in safeguarding crops and promoting healthier harvests with our another initiative of [Plant Disease Recognition](https://plantdiseaseprediction-jb29lbyqjxenfeg6lbfar7.streamlit.app/) System. */
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                ## 🗂️ About the Dataset

The dataset used for this model is from the [Fertilizer Prediction dataset](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction/data) available on Kaggle. It contains data on various factors affecting the choice of fertilizer for crops, including temperature, humidity, moisture, soil type, crop type, and the levels of nitrogen, potassium, and phosphorus. The target variable is the type of fertilizer recommended for the given conditions.

The dataset is valuable for building predictive models that can assist farmers in making informed decisions about fertilizer application to enhance crop yield and sustainability.








## 📁 About the ML Model

This machine learning model is designed to predict the appropriate fertilizer for crops based on various input features. The key steps involved in developing this model include:

### Data Preprocessing and EDA:

The dataset is read from a CSV file and inspected for basic information such as column names, unique values, and missing values.
Exploratory Data Analysis (EDA) is conducted to understand the distribution of continuous variables (Temperature, Humidity, Moisture, Nitrogen, Potassium, Phosphorus) using histograms and box plots, and to explore the relationship of categorical variables (Soil Type, Crop Type) with the target variable (Fertilizer) using count plots and box plots.
                
### Label Encoding:

Categorical variables (Soil_Type, Crop_Type, Fertilizer) are encoded into numerical values using LabelEncoder to prepare the data for modeling.
                
### Data Splitting:

The dataset is split into training and test sets using an 80/20 ratio.
                
### Model Training and Evaluation:

A RandomForestClassifier is trained on the training data.
GridSearchCV is used to tune hyperparameters (n_estimators, max_depth, min_samples_split) for the RandomForest model to improve performance.
The best model is evaluated on the test set, and its performance is assessed using classification metrics such as accuracy, classification report, and confusion matrix.
                """)

# Fertilizer Prediction Page[GitHub](https://github.com)
elif app_mode == "Fertilizer Prediction":
    st.header("Fertilizer Prediction")
    temp = st.slider("Temperature", min_value=0, max_value=100, step=1)
    humi = st.slider("Humidity", min_value=0, max_value=100, step=1)
    mois = st.slider("Moisture", min_value=0, max_value=100, step=1)
    soil = st.selectbox("Soil Type", ["Black", "Clayey", "Loamy", "Red", "Sandy"])
    crop = st.selectbox("Crop Type", ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil Seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"])
    nitro = st.slider("Nitrogen", min_value=0, max_value=100, step=1)
    pota = st.slider("Potassium", min_value=0, max_value=100, step=1)
    phosp = st.slider("Phosphorus", min_value=0, max_value=100, step=1)

    # Convert categorical inputs to numerical values
    soil_encoded = soil_mapping[soil]
    crop_encoded = crop_mapping[crop]

    # Predict button
    if st.button("Predict"):
        input_data = [int(temp), int(humi), int(mois), soil_encoded, crop_encoded, int(nitro), int(pota), int(phosp)]
        input_array = np.array(input_data).reshape(1, -1)
        result_index = classifier_model.predict(input_array)
        result_label = label_encoder.inverse_transform(result_index)
        st.success(f'Predicted Fertilizer is {result_label[0]}')
