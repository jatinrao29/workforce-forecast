#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- CONFIG ---
st.set_page_config(page_title="Staff Forecasting", layout="wide")

# --- TITLE ---
st.title("📈 Required Staff Forecast Dashboard")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("profit_data.csv")
    df.columns = df.columns.str.strip()
    df.set_index("month", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    return df

df = load_data()

# --- LOG TRANSFORM ---
df["required_staff_log"] = np.log(df["required_staff"])

# --- FORECAST HORIZON ---
st.sidebar.header("Forecast Settings")
forecast_steps = st.sidebar.selectbox("Select forecast horizon (months)", [3, 6, 9, 12], index=1)

# --- MODEL ---
train_data = df["required_staff_log"]
model = SARIMAX(train_data, order=(0, 1, 0), seasonal_order=(0, 1, 0, 12),
                enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(method='powell', disp=False)

# --- FORECAST ---
forecast_log = result.get_forecast(steps=forecast_steps)
forecast_log_mean = forecast_log.predicted_mean
forecast_log_conf_int = forecast_log.conf_int()

forecast = np.exp(forecast_log_mean)
forecast_conf_int = np.exp(forecast_log_conf_int)

# --- MERGE HISTORICAL + FORECAST FOR PLOTTING ---
combined_df = pd.concat([df["required_staff"], forecast], axis=0)
combined_df.name = "Required Staff"
actual = df["required_staff"]
predicted = forecast

# --- PLOT ---
fig, ax = plt.subplots(figsize=(12, 5))
actual.plot(ax=ax, label="Actual", color="orange")
predicted.plot(ax=ax, label="Forecast", color="green")
ax.fill_between(forecast_conf_int.index,
                forecast_conf_int.iloc[:, 0],
                forecast_conf_int.iloc[:, 1],
                color="lightgreen", alpha=0.3, label="Confidence Interval")
ax.set_title("Required Staff Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Staff Count")
ax.legend()
st.pyplot(fig)

# --- SHOW FORECAST TABLE ---
st.subheader("🔍 Forecasted Values")
forecast_table = pd.DataFrame({
    "Predicted": forecast,
    "Lower Bound": forecast_conf_int.iloc[:, 0],
    "Upper Bound": forecast_conf_int.iloc[:, 1]
})
st.dataframe(forecast_table)

# --- DOWNLOAD ---
csv = forecast_table.to_csv().encode("utf-8")
st.download_button("📥 Download Forecast as CSV", csv, file_name="staff_forecast.csv", mime="text/csv")


