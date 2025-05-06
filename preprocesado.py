import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Custom transformer for feature engineering
class PropertyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2025, n_location_clusters=5):
        self.current_year = current_year
        self.n_location_clusters = n_location_clusters
        self.kmeans = None
        # <-- we’ll fill this during fit
        self._feature_names_out = None            

    # ------------------------------------------------------------------ #
    def fit(self, X, y=None):
        coords = X[['lonxitude', 'latitude']].fillna(0)
        self.kmeans = KMeans(
            n_clusters=self.n_location_clusters,
            random_state=42
        ).fit(coords)

        # store the output column names once so we can reuse them later
        self._feature_names_out = np.array([
            'age', 'is_new', 'age_sq',
            'area_total', 'area_per_room', 'interior_ratio',
            'exterior_ratio', 'log_area_total', 'log_area_per_room',
            'bath_bed_ratio', 'rooms_per_m2',
            'prox_index', 'safe_prox',
            'has_exterior', 'trees_per_m2',
            'loc_cluster', 'build_decade',
            'crime_level', 'temp_zone',
            'orient_sin', 'orient_cos'
        ])

        return self

    def transform(self, X):
        X2 = X.copy()
        # Age features
        X2['age'] = self.current_year - X2['ano_construccion'].fillna(self.current_year)
        X2['is_new'] = (X2['age'] <= 5).astype(int)
        X2['age_sq'] = X2['age'] ** 2

        # Area features
        X2['area_total'] = X2['superficie_interior_m2'].fillna(0) + X2['superficie_exterior_m2'].fillna(0)
        X2['area_per_room'] = X2['area_total'] / X2['numero_habitacions'].replace(0, np.nan)
        X2['interior_ratio'] = X2['superficie_interior_m2'].fillna(0) / X2['area_total'].replace(0, np.nan)
        X2['exterior_ratio'] = X2['superficie_exterior_m2'].fillna(0) / X2['area_total'].replace(0, np.nan)
        X2['log_area_total'] = np.log1p(X2['area_total'])
        X2['log_area_per_room'] = np.log1p(X2['area_per_room'].fillna(0))

        # Density and ratios
        X2['bath_bed_ratio'] = X2['numero_banos'].fillna(0) / X2['numero_habitacions'].replace(0, np.nan)
        X2['rooms_per_m2'] = X2['numero_habitacions'].fillna(0) / X2['superficie_interior_m2'].replace(0, np.nan)

        # Proximity
        X2['prox_index'] = 1 / (1 + X2['distancia_centro_km'].fillna(0)) + 1 / (1 + X2['distancia_escola_km'].fillna(0))
        X2['safe_prox'] = X2['prox_index'] * (1 - X2['indice_criminalidade'].fillna(0))

        # Exterior and greenery
        X2['has_exterior'] = (X2['superficie_exterior_m2'].fillna(0) > 0).astype(int)
        X2['trees_per_m2'] = X2['numero_arboles_xardin'].fillna(0) / X2['area_total'].replace(0, np.nan)

        # Location clustering
        coords = X2[['lonxitude', 'latitude']].fillna(0)
        X2['loc_cluster'] = self.kmeans.predict(coords)

        # Categorical bins
        X2['build_decade'] = (X2['ano_construccion'].fillna(self.current_year) // 10) * 10
        X2['crime_level'] = pd.cut(
            X2['indice_criminalidade'].fillna(0),
            bins=[-np.inf, 0.2, 0.5, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        X2['temp_zone'] = pd.cut(
            X2['temperatura_media_mes_construccion'].fillna(X2['temperatura_media_mes_construccion'].median()),
            bins=[-np.inf, 10, 15, 20, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Orientation cyclic
        orient_map = {'Norte': 0, 'Este': 90, 'Sur': 180, 'Oeste': 270}
        angles = X2['orientacion'].map(orient_map).fillna(0)
        rad = np.deg2rad(angles)
        X2['orient_sin'] = np.sin(rad)
        X2['orient_cos'] = np.cos(rad)

        # Select features
        features = [
            'age', 'is_new', 'age_sq', 'area_total', 'area_per_room', 'interior_ratio',
            'exterior_ratio', 'log_area_total', 'log_area_per_room', 'bath_bed_ratio',
            'rooms_per_m2', 'prox_index', 'safe_prox', 'has_exterior', 'trees_per_m2',
            'loc_cluster', 'build_decade', 'crime_level', 'temp_zone', 'orient_sin', 'orient_cos'
        ]
        Xf = X2[features]
        return Xf
    
    # ------------------------------------------------------------------ #
    def get_feature_names_out(self, input_features=None):
        """
        Called by ColumnTransformer to learn this
        transformer’s output column names.
        """
        return self._feature_names_out

# Column lists
numeric_feats = [
    'superficie_interior_m2', 'superficie_exterior_m2', 'numero_habitacions',
    'numero_banos', 'ano_construccion', 'temperatura_media_mes_construccion',
    'distancia_centro_km', 'distancia_escola_km', 'indice_criminalidade',
    'numero_arboles_xardin'
]
ord_feats = ['calidade_materiais', 'acceso_transporte_publico', 'eficiencia_enerxetica']
ord_cats = [
    ['Baixa', 'Media', 'Alta'],
    ['Malo', 'Regular', 'Bo'],
    ['G', 'F', 'E', 'D', 'C', 'B', 'A']
]
nom_feats = ['tipo_edificacion', 'cor_favorita_propietario', 'orientacion']

# Preprocessing transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ord_cats))
])
nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ('prop_feats', PropertyFeatures(), [
        'superficie_interior_m2', 'superficie_exterior_m2', 'numero_habitacions', 'numero_banos',
        'ano_construccion', 'lonxitude', 'latitude', 'temperatura_media_mes_construccion',
        'distancia_centro_km', 'distancia_escola_km', 'indice_criminalidade', 'numero_arboles_xardin',
        'orientacion'
    ]),
    ('num', numeric_transformer, numeric_feats),
    ('ord', ordinal_transformer, ord_feats),
    ('nom', nominal_transformer, nom_feats)
], remainder='drop')

# Load data
df_train = pd.read_csv('train.csv')
X_train = df_train.drop(columns=['id', 'prezo_euros'])
y_train = df_train['prezo_euros']

df_test = pd.read_csv('test.csv')
X_test = df_test.drop(columns=['id'])

# Fit and transform
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Convert to DataFrame
# Note: Feature names can be retrieved via preprocessor.get_feature_names_out()
train_cols = preprocessor.get_feature_names_out()
df_train_feats = pd.DataFrame(X_train_trans, columns=train_cols)
df_train_feats.insert(0, 'prezo_euros', y_train.values)
df_test_feats = pd.DataFrame(X_test_trans, columns=train_cols)

# Save to CSV
df_train_feats.to_csv('train_features.csv', index=False)
df_test_feats.to_csv('test_features.csv', index=False)

print('Features saved: train_features.csv, test_features.csv')
