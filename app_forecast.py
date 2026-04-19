import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Production Forecasting", layout="centered")

st.title("📊 Production Forecasting App")
st.markdown("Forecast future production using Time Series (ARIMA)")

st.divider()

# -------- GENERATE DATA --------
date = pd.date_range(start='2022-01-01', periods=200)
production = np.random.randint(80, 120, size=200)

df = pd.DataFrame({
    'Date': date,
    'Production': production
})

df.set_index('Date', inplace=True)

# -------- SHOW DATA --------
st.subheader("📈 Historical Production")
st.line_chart(df)

# -------- USER INPUT --------
days = st.slider("Select number of days to forecast", 1, 30, 10)

# -------- MODEL --------
model = ARIMA(df['Production'], order=(1,1,1))
model_fit = model.fit()

# -------- FORECAST --------
forecast = model_fit.forecast(steps=days)

# -------- OUTPUT --------
st.subheader("🔮 Forecasted Production")
st.write(forecast)

# -------- PLOT --------
fig, ax = plt.subplots()
df['Production'].plot(ax=ax, label='Actual')
forecast.plot(ax=ax, label='Forecast')
plt.legend()

st.pyplot(fig)

st.divider()

st.markdown("Built using Streamlit & ARIMA 🚀")