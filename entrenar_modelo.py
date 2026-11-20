from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'dataset_agro_clima_con_ratio_perdida.csv'
MODEL_PATH = BASE_DIR / 'modelo_agro.joblib'


def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
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

    model_pipeline.fit(X[train_mask], y[train_mask])
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f'Modelo guardado en: {MODEL_PATH}')


if __name__ == '__main__':
    main()
