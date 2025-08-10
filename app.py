import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Load models and encoders
lr = joblib.load("linear_regression_model.pkl")
rf = joblib.load("random_forest_model.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("year_scaler.pkl")

# Load original data for plotting
df = pd.read_csv("rainfall.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df['SUBDIVISION_ENC'] = le.transform(df['SUBDIVISION'])

# UI Title
st.title("🌧️ India Rainfall Prediction Dashboard")

# Sidebar Inputs
subdiv = st.selectbox("Select Subdivision", sorted(df['SUBDIVISION'].unique()))
model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
predict_type = st.radio("Predict for", ["Monthly", "Seasonal", "Annual"])
year = st.number_input("Enter Year to Predict", min_value=2016, max_value=2100, value=2025)

if predict_type == "Monthly":
    month = st.selectbox("Select Month", ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
    month_num = datetime.strptime(month, "%b").month
    st.write(f"Predicting {month} rainfall in {year} for {subdiv}")
    
    X = pd.DataFrame([[le.transform([subdiv])[0], year, month_num]], columns=["SUBDIVISION_ENC", "YEAR", "MONTH_NUM"])
    model = lr if model_name == "Linear Regression" else rf
    prediction = model.predict(X)[0]
    st.metric(label="Predicted Rainfall (mm)", value=f"{prediction:.2f}")

    # Historical plot
    hist = df[df['SUBDIVISION'] == subdiv][['YEAR', month]].dropna()
    st.line_chart(hist.rename(columns={month: 'Rainfall'}).set_index('YEAR'))

elif predict_type == "Seasonal":
    season = st.selectbox("Select Season", ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'])
    st.write(f"Predicting {season} rainfall in {year} for {subdiv}")
    
    X = pd.DataFrame([[le.transform([subdiv])[0], year, 0]], columns=["SUBDIVISION_ENC", "YEAR", "MONTH_NUM"])
    model = lr if model_name == "Linear Regression" else rf

    # Get average of seasonal rainfall over the years as a proxy for month_num=0
    prediction = model.predict(X)[0]  # rough estimate
    st.metric(label="Predicted Rainfall (mm)", value=f"{prediction:.2f}")
    
    hist = df[df['SUBDIVISION'] == subdiv][['YEAR', season]].dropna()
    st.line_chart(hist.rename(columns={season: 'Rainfall'}).set_index('YEAR'))

else:  # Annual
    st.write(f"Predicting Annual rainfall in {year} for {subdiv}")
    X = pd.DataFrame([[le.transform([subdiv])[0], year, 0]], columns=["SUBDIVISION_ENC", "YEAR", "MONTH_NUM"])
    model = lr if model_name == "Linear Regression" else rf
    prediction = model.predict(X)[0]  # estimate
    st.metric(label="Predicted Annual Rainfall (mm)", value=f"{prediction:.2f}")

    hist = df[df['SUBDIVISION'] == subdiv][['YEAR', 'ANNUAL']].dropna()
    st.line_chart(hist.rename(columns={'ANNUAL': 'Rainfall'}).set_index('YEAR'))
