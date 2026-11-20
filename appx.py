import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

# Configuración inicial de la página web
st.set_page_config(
    page_title="Predictor de Rendimiento Agrícola", page_icon="🌾", layout="centered"
)

st.title("🌾 Aplicativo Predictivo de Rendimiento Agrícola")
st.markdown(
    "Herramienta inteligente basada en **Machine Learning (Random Forest "
    "Optimizado)**. Ingrese la ubicación y el cultivo; el sistema estimará el "
    "rinde basándose en los patrones climáticos históricos de la zona."
)


@st.cache_resource
def entrenar_modelo_campeon():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  csv_path = os.path.join(BASE_DIR, 'dataset_agro_clima_con_ratio_perdida.csv')

  df = pd.read_csv(csv_path, encoding='utf-8-sig')
  df_model = df[df['rendimiento_kgxha'] > 0].copy()

  features_num = [
      'anio',
      'lluvia_total_ciclo_mm',
      'lluvia_periodo_critico_mm',
      'temp_media_ciclo',
      'dias_ola_calor',
      'ratio_perdida_area',
  ]
  features_cat = ['cultivo', 'provincia', 'grupo_cultivo']

  X = df_model[features_num + features_cat]
  y = df_model['rendimiento_kgxha']

  train_mask = X['anio'] <= 2018
  X_train, y_train = X[train_mask], y[train_mask]

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), features_num),
          (
              'cat',
              OneHotEncoder(handle_unknown='ignore', sparse_output=False),
              features_cat,
          ),
      ]
  )

  model_pipeline = Pipeline(
      steps=[
          ('preprocessor', preprocessor),
          (
              'model',
              RandomForestRegressor(
                  n_estimators=100,
                  max_depth=15,
                  min_samples_split=10,
                  random_state=42,
                  n_jobs=-1,
              ),
          ),
      ]
  )

  model_pipeline.fit(X_train, y_train)
  return model_pipeline, df


with st.spinner('Cargando y optimizando el motor de Machine Learning...'):
  pipeline, df_base = entrenar_modelo_campeon()

st.success('¡Modelo cargado con éxito!')

# --- INTERFAZ SIMPLIFICADA (MENOS FRICCIÓN PARA EL USUARIO) ---
st.sidebar.header('Ubicación y Cultivo')

cultivos_disponibles = sorted(df_base['cultivo'].unique())
cultivo_sel = st.sidebar.selectbox('Seleccione el Cultivo', cultivos_disponibles)

grupo_cultivo_sel = df_base.loc[
    df_base['cultivo'] == cultivo_sel, 'grupo_cultivo'
].values[0]
st.sidebar.info(f'**Grupo Agronómico:** {grupo_cultivo_sel.capitalize()}')

provincias_disponibles = sorted(df_base['provincia'].unique())
provincia_sel = st.sidebar.selectbox('Provincia', provincias_disponibles)

departamentos_disponibles = sorted(
    df_base[df_base['provincia'] == provincia_sel]['departamento'].unique()
)
departamento_sel = st.sidebar.selectbox(
    'Departamento / Partido', departamentos_disponibles
)

anio_sel = st.sidebar.slider(
    'Año de Campaña Objetivo', min_value=2019, max_value=2030, value=2026, step=1
)

# Opcional: Permitir simular escenarios climáticos generales
st.sidebar.subheader('Escenario Climático Zonal')
escenario_climatico = st.sidebar.selectbox(
    'Condición Hídrica Esperada',
    ['Promedio Histórico (Normal)', 'Año Seco (Defícit)', 'Año Húmedo (Niño)'],
)

# --- LÓGICA DE IMPUTACIÓN AUTOMÁTICA DE CLIMA BASADA EN HISTÓRICO ZONAL ---
# Filtramos el histórico de ese departamento y cultivo para calcular la mediana climática real
df_zonal = df_base[
    (df_base['provincia'] == provincia_sel)
    & (df_base['departamento'] == departamento_sel)
    & (df_base['cultivo'] == cultivo_sel)
]

if df_zonal.empty:
  # Fallback global si no hay datos específicos para ese departamento exacto
  df_zonal = df_base[df_base['cultivo'] == cultivo_sel]

# Calculamos las medianas históricas base
base_lluvia_tot = df_zonal['lluvia_total_ciclo_mm'].median()
base_lluvia_crit = df_zonal['lluvia_periodo_critico_mm'].median()
base_temp = df_zonal['temp_media_ciclo'].median()
base_calor = int(df_zonal['dias_ola_calor'].median())

# Ajustamos según el escenario seleccionado por el usuario
if escenario_climatico == 'Año Seco (Defícit)':
  base_lluvia_tot *= 0.75
  base_lluvia_crit *= 0.70
  base_calor += 5
elif escenario_climatico == 'Año Húmedo (Niño)':
  base_lluvia_tot *= 1.25
  base_lluvia_crit *= 1.20
  base_calor = max(0, base_calor - 2)

# Ratio de pérdida típico por defecto
ratio_perdida_def = 0.05

# --- PREDICCIÓN AUTOMÁTICA ---
if st.button('Estimar Rendimiento de la Región'):
  input_data = pd.DataFrame({
      'anio': [anio_sel],
      'lluvia_total_ciclo_mm': [base_lluvia_tot],
      'lluvia_periodo_critico_mm': [base_lluvia_crit],
      'temp_media_ciclo': [base_temp],
      'dias_ola_calor': [base_calor],
      'ratio_perdida_area': [ratio_perdida_def],
      'cultivo': [cultivo_sel],
      'provincia': [provincia_sel],
      'grupo_cultivo': [grupo_cultivo_sel],
  })

  prediccion_rinde = pipeline.predict(input_data)[0]

  st.markdown('---')
  st.subheader('Resultado de la Predicción')
  col1, col2 = st.columns(2)

  with col1:
    st.metric(
        label='Rendimiento Estimado', value=f'{prediccion_rinde:,.2f} kg/ha'
    )

  with col2:
    st.metric(
        label='Potencial Productivo',
        value=f'{(prediccion_rinde / 1000):.2f} tn/ha',
    )

  st.success(
      f'**Simulación exitosa:** Para el cultivo de **{cultivo_sel}** en el'
      f' departamento de **{departamento_sel} ({provincia_sel})** bajo un'
      f' escenario climático **{escenario_climatico}** en la campaña'
      f' **{anio_sel}**, el modelo estima un rinde de **{prediccion_rinde:,.2f}'
      f' kg/ha** (utilizando una referencia histórica zonal de'
      f' {base_lluvia_crit:.0f} mm críticos).'
  )
