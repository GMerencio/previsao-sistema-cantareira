import pandas as pd
import streamlit as st
import datetime as dt
from prophet import Prophet
from prophet.serialize import model_from_json
from plotting_utils import plot_plotly
from st_pages import Page, show_pages

# Configurações da página

st.set_page_config(
	page_title='Previsão de volume do Sistema Cantareira'
)

# Salvar dataframe em formatos distintos. Valores permitidos para o
# formato: 'csv', 'parquet', 'json'

@st.cache_data
def converter_df(df, formato):
	if formato == 'csv':
		return df.to_csv().encode('utf-8')
	if formato == 'parquet':
		return df.to_parquet()
	if formato == 'json':
		return df.to_json()

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
	
# Definição das páginas da aplicação

show_pages(
    [
        Page('streamlit-app.py', 'Página inicial', ':house:'),
        Page('pages/dados.py', 'Dados brutos', ':bar_chart:'),
        Page('pages/info_modelo.py', 'Sobre o modelo', ':books:'),
    ]
)

# Formata uma data no esquema dia/mês/ano, recebendo como entrada
# um objeto pandas.Timestamp

def formatar_data(pd_data):
	return pd_data.strftime('%m/%Y')
	
# Interface

st.title('Previsão de volume do Sistema Cantareira')

modeloVolProphet = carregar_modeloVolProphet()
previsaoVolProphet = carregar_previsaoVolProphet()
modeloChuvaProphet = carregar_modeloChuvaProphet()
previsaoChuvaProphet = carregar_previsaoChuvaProphet()

st.header('Volume (em hm³)')
st.plotly_chart(plot_plotly(modeloVolProphet, previsaoVolProphet, xlabel='Data', ylabel='Volume (hm³)'))
dadosPrevisaoVol = previsaoVolProphet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
vol_csv = converter_df(dadosPrevisaoVol, 'csv')
vol_parquet = converter_df(dadosPrevisaoVol, 'parquet')
vol_json = converter_df(dadosPrevisaoVol, 'json')

st.download_button(
	label="Baixar os dados como .csv",
	data=vol_csv,
	file_name='dadosPrevisaoVol.csv',
	mime='text/csv'
)

st.download_button(
	label="Baixar os dados como .parquet",
	data=vol_parquet,
	file_name='dadosPrevisaoVol.parquet',
	mime='application/octet-stream'
)

st.download_button(
	label="Baixar os dados como .json",
	data=vol_json,
	file_name='dadosPrevisaoVol.json',
	mime='application/json'
)

st.header('Chuva (em mm)')
st.plotly_chart(plot_plotly(modeloChuvaProphet, previsaoChuvaProphet, xlabel='Data', ylabel='Chuva (mm)'))
dadosPrevisaoChuva = previsaoVolProphet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
chuva_csv = converter_df(dadosPrevisaoChuva, 'csv')
chuva_parquet = converter_df(dadosPrevisaoChuva, 'parquet')
chuva_json = converter_df(dadosPrevisaoChuva, 'json')

st.download_button(
	label="Baixar os dados como .csv",
	data=chuva_csv,
	file_name='dadosPrevisaoChuva.csv',
	mime='text/csv'
)

st.download_button(
	label="Baixar os dados como .parquet",
	data=chuva_parquet,
	file_name='dadosPrevisaoChuva.parquet',
	mime='application/octet-stream'
)

st.download_button(
	label="Baixar os dados como .json",
	data=chuva_json,
	file_name='dadosPrevisaoChuva.json',
	mime='application/json'
)