import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# Importar dados
df_SABESP = pd.read_excel('../SABESP-sistemas_produtores.xlsx')

# Agrupar dados para obter previsões do Sistema Cantareira como um todo
dfSabesp = df_SABESP.groupby(['Data'], as_index=False).sum(numeric_only=True)

# Criar dataframe para utilizar com o Prophet
dfProphet = pd.DataFrame()
dfProphet['ds'] = dfSabesp['Data']
dfProphet['y'] = dfSabesp['Volume (hm³)']

# Ajustar modelo
modeloVol = Prophet()
modeloVol.fit(dfProphet)

# Gerar modelo
DataFrameFuturoVol = modeloVol.make_future_dataframe(periods=730, freq = 'd')
previsaoVol = modeloVol.predict(DataFrameFuturoVol)

# Salvar modelo
with open('modelo-volume.json', 'w') as file:
	file.write(model_to_json(modeloVol))

# Salvar dataframe gerado pelo modelo
previsaoVol.to_pickle('previsao-volume.pkl')