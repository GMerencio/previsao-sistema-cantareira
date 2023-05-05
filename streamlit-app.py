import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.serialize import model_from_json
from plotting_utils import plot_plotly

# Carregar modelos

@st.cache_data
def carregar_modeloVolProphet():
	with open('modelo-prophet/modelo-volume.json', 'r') as file:
		return model_from_json(file.read())

@st.cache_data
def carregar_modeloChuvaProphet():
	with open('modelo-prophet/modelo-chuva.json', 'r') as file:
		return model_from_json(file.read())

# Carregar dados previstos

@st.cache_data
def carregar_previsaoVolProphet():
	return pd.read_pickle('modelo-prophet/previsao-volume.pkl')

@st.cache_data
def carregar_previsaoChuvaProphet():
	return pd.read_pickle('modelo-prophet/previsao-chuva.pkl')
	
# Interface

st.title('Previsão de volume do Sistema Cantareira')

modeloVolProphet = carregar_modeloVolProphet()
previsaoVolProphet = carregar_previsaoVolProphet()
modeloChuvaProphet = carregar_modeloChuvaProphet()
previsaoChuvaProphet = carregar_previsaoChuvaProphet()

st.header('Volume (em hm³)')
st.plotly_chart(plot_plotly(modeloVolProphet, previsaoVolProphet))
#st.write(previsaoVolProphet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

st.header('Chuva (em mm)')
st.plotly_chart(plot_plotly(modeloChuvaProphet, previsaoChuvaProphet))