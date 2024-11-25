import streamlit as st
import pandas as pd
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

# Load dataset
data = pd.read_csv("C:\\Users\\rejas\\Desktop\\Prakash Senapati sir\\Prakash Senapati lab\\22.11.24\\new_dataset.csv")

# Define independent variables and dependent variable
X = data[['bedrooms', 'bathrooms', 'floors', 'sqft_lot15']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Real Estate Price Predictor')

st.write('Enter the details below to predict the price of the flat.')

# Input fields for user to enter details
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
floors = st.number_input('Number of Floors', min_value=1, max_value=5, value=1)
sqft_lot15 = st.number_input('Square Feet', min_value=500, max_value=1000000, value=2000)

# Make prediction
input_data = pd.DataFrame([[bedrooms, bathrooms, floors, sqft_lot15]], columns=['bedrooms', 'bathrooms', 'floors', 'sqft_lot15'])

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    # Display prediction
    st.write(f'The predicted price of the flat is: ${prediction:.2f}')

