import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

## streamlit app
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or not based on their information.")
st.write("Please enter the customer information below:")
# Input fields
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geography_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geography_encoded_df], axis=1)
input_data = input_data.drop(columns=['Geography'])
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

st.write(f"The probability of churn is {prediction[0][0]:.2f}")
if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")