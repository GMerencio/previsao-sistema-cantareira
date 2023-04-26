import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.express as px

st.title('Previsão de volume do Sistema Cantareira')
df_SABESP = pd.read_excel('../SABESP-sistemas_produtores.xlsx')
dfSabesp = df_SABESP.groupby(['Data'], as_index=False).sum(numeric_only=True)

dfProphet = pd.DataFrame()
dfProphet['ds'] = dfSabesp['Data']
dfProphet['y'] = dfSabesp['Volume (hm³)']

modeloVol = Prophet()
modeloVol.fit(dfProphet)
DataFrameFuturoVol = modeloVol.make_future_dataframe(periods=730, freq = 'd')
previsaoVol = modeloVol.predict(DataFrameFuturoVol)

previsaoVolIntervalo = previsaoVol[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
st.write(previsaoVolIntervalo)

figure = modeloVol.plot(previsaoVol)
st.write(figure)