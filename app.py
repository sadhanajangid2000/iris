# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

# UI
st.title("Iris Flower Prediction App")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width(cm)'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"The predicted Iris class is: {prediction[0]}")
