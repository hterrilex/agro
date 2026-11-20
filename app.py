import os
import pandas as pd
import joblib
import streamlit as st

# Configuración inicial de la página web
st.set_page_config(
    page_title="Predictor de Rendimiento Agrícola", page_icon="🌾", layout="centered"
)

st.title("🌾 Aplicativo Predictivo de Rendimiento Agrícola")
st.markdown(
    "Herramienta inteligente basada en **Machine Learning (Random Forest"
    " Optimizado)**. Ingrese la ubicación y el cultivo; el sistema estimará el"
    " rinde basándose en los patrones climáticos históricos de la zona."
)


@st.cache_resource
def cargar_modelo():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(BASE_DIR, 'modelo_agro.joblib')

  if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"No se encontró el modelo preentrenado en: {model_path}. "
        "Ejecute entrenar_modelo.py para generarlo."
    )

  return joblib.load(model_path)


@st.cache_data
def cargar_datos():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  csv_path = os.path.join(BASE_DIR, 'dataset_agro_clima_con_ratio_perdida.csv')

  if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {csv_path}")

  return pd.read_csv(csv_path, encoding='utf-8-sig')


with st.spinner('Cargando modelo preentrenado y datos...'):
  pipeline = cargar_modelo()
  df_base = cargar_datos()

st.success('¡Aplicación lista para usar!')

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
  col1, col2 = col3 = st.columns(2)

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
      f' escenario climáico **{escenario_climatico}** en la campaña'
      f' **{anio_sel}**, el modelo estima un rinde de **{prediccion_rinde:,.2f}'
      f' kg/ha** (utilizando una referencia histórica zonal de'
      f' {base_lluvia_crit:.0f} mm críticos).'
  )