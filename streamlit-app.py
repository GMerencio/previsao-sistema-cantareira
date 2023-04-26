import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly

st.title('Previsão de volume do Sistema Cantareira')

modeloVolProphet = None
with open('modelo-prophet/modelo-volume.json', 'r') as file:
	modeloVolProphet = model_from_json(file.read())
previsaoVolProphet = pd.read_pickle('modelo-prophet/previsao-volume.pkl')

st.write(previsaoVolProphet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
st.plotly_chart(plot_plotly(modeloVolProphet, previsaoVolProphet))