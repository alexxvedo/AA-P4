
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet


###############################################################################
# 1. Extended Feature Engineering
###############################################################################
class ExtendedFeatureEngineer(BaseEstimator, TransformerMixin):
    _orientation_map = {
        "Norte": 0, "Nordeste": 45, "Este": 90, "Sudeste": 135,
        "Sur": 180, "Sudoeste": 225, "Oeste": 270, "Noroeste": 315,
    }

    def __init__(self, current_year: int = 2025, geo_clusters: int = 10):
        self.current_year = current_year
        self.geo_clusters = geo_clusters
        self.km_model_ = None
        self.agg_stats_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # Fit KMeans on latitude & longitude
        if {"latitud", "longitud"}.issubset(X.columns):
            coords = X[["latitud", "longitud"]].fillna(0)
            self.km_model_ = KMeans(n_clusters=self.geo_clusters, random_state=42)
            self.km_model_.fit(coords)
        # Compute aggregate stats by building type if target y provided
        if y is not None and 'tipo_edificacion' in X.columns:
            grouped = pd.DataFrame({'precio': y, 'type': X['tipo_edificacion']})
            agg = grouped.groupby('type').precio.agg(['mean', 'std'])
            self.agg_stats_ = agg.to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        # Age features
        df['antiguedad'] = self.current_year - df['ano_construccion']
        df['antiguedad2'] = df['antiguedad'] ** 2
        df['decada'] = (df['ano_construccion'] // 10) * 10

        # Surface features
        df['superficie_total'] = df['superficie_interior_m2'].fillna(0) + df['superficie_exterior_m2'].fillna(0)
        df['log_superficie_total'] = np.log1p(df['superficie_total'])
        df['habitacion_area'] = df['superficie_interior_m2'] / df['numero_habitacions'].replace(0, np.nan)
        df['banos_area'] = df['superficie_interior_m2'] / df['numero_banos'].replace(0, np.nan)

        # Distance transformations
        for col in ['distancia_centro_km', 'distancia_escola_km']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])
                df[f'inv_{col}'] = 1 / (df[col] + 0.1)

        # Temperature features
        if 'temperatura_media_mes_construccion' in df.columns:
            temp = df['temperatura_media_mes_construccion']
            df['temp_norm'] = (temp - temp.mean()) / temp.std()
            df['temp_sq'] = temp ** 2

        # Crime index buckets
        if 'indice_criminalidade' in df.columns:
            df['crime_q'] = pd.qcut(df['indice_criminalidade'], 5, labels=False, duplicates='drop')

        # Orientation encoding
        deg = df.get('orientacion', pd.Series()).map(self._orientation_map).fillna(0)
        rad = np.deg2rad(deg)
        df['orient_sin'] = np.sin(rad)
        df['orient_cos'] = np.cos(rad)

        # Geo clusters
        if self.km_model_ is not None:
            coords = df[['latitud', 'longitud']].fillna(0)
            df['geo_cluster'] = self.km_model_.predict(coords)
        else:
            df['geo_cluster'] = 0

        # Aggregated stats by building type
        if 'tipo_edificacion' in df.columns and self.agg_stats_:
            df['type_price_mean'] = df['tipo_edificacion'].map(self.agg_stats_['mean'])
            df['type_price_std'] = df['tipo_edificacion'].map(self.agg_stats_['std'])
        else:
            df['type_price_mean'] = 0
            df['type_price_std'] = 0

        # One-hot favorite color if exists
        if 'cor_favorita_propietario' in df.columns:
            colors = pd.get_dummies(df['cor_favorita_propietario'], prefix='color')
            df = pd.concat([df, colors], axis=1)

        # Date features
        if 'fecha' in df.columns:
            dt = pd.to_datetime(df['fecha'], errors='coerce')
            df['mes'] = dt.dt.month
            df['dia'] = dt.dt.day
            df['dia_semana'] = dt.dt.weekday
            df['is_fin_de_semana'] = dt.dt.weekday.isin([5, 6]).astype(int)

        # Drop original columns
        drops = [
            'ano_construccion', 'superficie_interior_m2', 'superficie_exterior_m2',
            'numero_habitacions', 'numero_banos', 'temperatura_media_mes_construccion',
            'distancia_centro_km', 'distancia_escola_km', 'indice_criminalidade',
            'orientacion', 'tipo_edificacion', 'cor_favorita_propietario', 'fecha'
        ]
        df.drop(columns=[c for c in drops if c in df.columns], inplace=True)
        return df

###############################################################################
# 2. Winsorizer Selectivo
###############################################################################
class WinsorizerSelective(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.005, upper=0.995, active=True):
        self.lower = lower
        self.upper = upper
        self.active = active
        self.bounds_ = {}

    def fit(self, X, y=None):
        if not self.active:
            return self
        df = pd.DataFrame(X)
        for col in df.columns:
            lo = df[col].quantile(self.lower)
            hi = df[col].quantile(self.upper)
            self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X):
        if not self.active:
            return X
        df = pd.DataFrame(X).copy()
        for col, (lo, hi) in self.bounds_.items():
            df[col] = df[col].clip(lo, hi)
        return df.values

###############################################################################
# 3. Build Preprocessor
###############################################################################
def build_preprocessor(df_sample: pd.DataFrame, is_train=True):
    target = 'prezo_euros'

    # Aplica la ingeniería de características para ver columnas reales
    feature_engineer = ExtendedFeatureEngineer()
    df_engineered = feature_engineer.fit_transform(df_sample.copy(), df_sample[target] if target in df_sample else None)

    # Detectar columnas después de la transformación
    numeric_cols = df_engineered.select_dtypes(include='number').columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    categorical_cols = df_engineered.select_dtypes(include='object').columns.tolist()

    # Ordinal mappings
    ordinal_maps = {
        'calidade_materiais': ['Baixa', 'Media', 'Alta'],
        'acceso_transporte_publico': ['Malo', 'Regular', 'Bo', 'Moi bo'],
        'eficiencia_enerxetica': ['G', 'F', 'E', 'D', 'C', 'B', 'A'],
    }
    ord_feats = [c for c in ordinal_maps if c in categorical_cols]
    ord_cats = [ordinal_maps[c] for c in ord_feats]
    onehot_feats = [c for c in categorical_cols if c not in ord_feats]

    # Pipelines
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('winsor', WinsorizerSelective(active=is_train)),
        ('scale', RobustScaler())
    ])
    ord_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(categories=ord_cats))
    ])
    ohe_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('ord', ord_pipe, ord_feats),
        ('ohe', ohe_pipe, onehot_feats)
    ], remainder='drop', n_jobs=-1)

    # Full pipeline
    full_pipeline = Pipeline([
        ('features', feature_engineer),
        ('preproc', preprocessor)
    ])

    return full_pipeline, target

###############################################################################
# 4. Uso para train/test
###############################################################################
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    # Split antes del preprocesamiento para evitar leakage
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # Construir preprocesador con solo datos de entrenamiento
    train_pipe, target = build_preprocessor(df_train, is_train=True)
    X_train = train_pipe.fit_transform(df_train, df_train[target])
    y_train = df_train[target].values

    # Transformar validación y test sin ajustar de nuevo
    feat_eng = train_pipe.named_steps['features']
    preproc = train_pipe.named_steps['preproc']

    X_val = preproc.transform(feat_eng.transform(df_val))
    y_val = df_val[target].values

    X_test = preproc.transform(feat_eng.transform(df_test))

    # Obtener nombres de columnas como antes
    column_names = []
    for name, transformer, cols in preproc.transformers_:
        if transformer == 'drop':
            continue
        elif hasattr(transformer, 'get_feature_names_out'):
            try:
                column_names.extend(transformer.get_feature_names_out(cols))
            except:
                column_names.extend(cols)
        else:
            column_names.extend(cols)

    # Guardar como CSV
    pd.DataFrame(X_train, columns=column_names).assign(prezo_euros=y_train).to_csv('train_preprocesado.csv', index=False)
    pd.DataFrame(X_val, columns=column_names).assign(prezo_euros=y_val).to_csv('val_preprocesado.csv', index=False)
    pd.DataFrame(X_test, columns=column_names).to_csv('test_preprocesado.csv', index=False)
